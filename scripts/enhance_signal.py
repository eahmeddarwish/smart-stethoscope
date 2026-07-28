# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
"""
Heart Sound Signal Enhancement Pipeline
Smart Stethoscope Project

Pipeline:
  Record 6s @ 44100 Hz
  -> Downsample to 2000 Hz
  -> Bandpass (20-950 Hz) + Notch (50 Hz)
  -> Noise Gate (remove low-energy segments)
  -> Beat Detection (find heartbeat peaks)
  -> Best 3s Window Selection (highest SNR segment)
  -> Save before/after comparison + WAV files
"""

import numpy as np
import sounddevice as sd
import scipy.signal as sig
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import datetime

# ============================================================
# SETTINGS
# ============================================================
RECORD_SEC  = 8          # record longer for better beat picking
TARGET_SR   = 2000       # must match model training
CAPTURE_SR  = 44100      # USB mic native rate
CLIP_SEC    = 3          # model expects exactly 3 seconds
SAMPLES     = TARGET_SR * CLIP_SEC
DEVICE      = None       # None = system default (USB mic)
OUTPUT_DIR  = os.path.expanduser("~/Desktop/Smart_Stethoscope")

# Noise gate threshold (fraction of max RMS)
# frames below this are considered noise-only
NOISE_GATE  = 0.30

# ============================================================
# STEP 1: RECORD
# ============================================================
def record(duration=RECORD_SEC):
    print(f"\n[1] Recording {duration}s @ {CAPTURE_SR} Hz ...")
    print("    >>> ضع السماعة على صدرك بثبات <<<")
    os.system("amixer -c 3 sset Mic 30% > /dev/null 2>&1")
    raw = sd.rec(int(duration * CAPTURE_SR),
                 samplerate=CAPTURE_SR, channels=1,
                 dtype='float32', device=DEVICE)
    sd.wait()
    raw = raw.flatten()
    print(f"    Done. Peak={np.max(np.abs(raw)):.4f}")
    return raw

# ============================================================
# STEP 2: DOWNSAMPLE
# ============================================================
def downsample(audio_44k):
    factor = CAPTURE_SR // TARGET_SR
    audio_2k = sig.decimate(audio_44k, factor, ftype='fir', zero_phase=True)
    return audio_2k.astype(np.float32)

# ============================================================
# STEP 3: BANDPASS + NOTCH FILTER
# ============================================================
def apply_filters(audio):
    nyq  = TARGET_SR / 2.0
    # Bandpass 20-950 Hz (Nyquist-safe)
    low  = 20.0 / nyq
    high = min(0.95, 950.0 / nyq)
    b, a = sig.butter(4, [low, high], btype='band')
    filtered = sig.filtfilt(b, a, audio.astype(np.float64))
    # Notch 50 Hz
    b_n, a_n = sig.iirnotch(50.0, 30, TARGET_SR)
    filtered  = sig.filtfilt(b_n, a_n, filtered)
    return filtered.astype(np.float32)

# ============================================================
# STEP 4: NOISE GATE
# ============================================================
def noise_gate(audio, frame_sec=0.05, gate_ratio=NOISE_GATE):
    """
    Zero out frames where RMS < gate_ratio * max_RMS
    Keeps only the loud parts (actual heartbeat beats)
    """
    frame_len = int(TARGET_SR * frame_sec)
    hop       = frame_len // 2
    output    = audio.copy()

    # Compute RMS per frame
    rms_vals = []
    positions = []
    for i in range(0, len(audio) - frame_len, hop):
        frame_rms = np.sqrt(np.mean(audio[i:i+frame_len]**2))
        rms_vals.append(frame_rms)
        positions.append(i)

    rms_vals = np.array(rms_vals)
    threshold = gate_ratio * np.max(rms_vals)

    # Zero out below-threshold frames
    gated_mask = np.zeros(len(audio), dtype=bool)
    for idx, i in enumerate(positions):
        if rms_vals[idx] >= threshold:
            end = min(i + frame_len, len(audio))
            gated_mask[i:end] = True

    output[~gated_mask] = 0.0

    kept_pct = 100.0 * np.sum(gated_mask) / len(audio)
    print(f"    Noise gate: kept {kept_pct:.1f}% of signal")
    return output, gated_mask

# ============================================================
# STEP 5: BEST 3s WINDOW SELECTION
# ============================================================
def find_best_window(audio, win_sec=CLIP_SEC):
    """
    Slide a 3s window over the signal
    Pick the window with highest RMS energy
    (= most heartbeat content, least silence)
    """
    win_len  = int(TARGET_SR * win_sec)
    hop      = int(TARGET_SR * 0.1)   # 100ms step

    best_rms   = -1
    best_start = 0

    for start in range(0, len(audio) - win_len, hop):
        window = audio[start:start + win_len]
        rms = np.sqrt(np.mean(window**2))
        if rms > best_rms:
            best_rms   = rms
            best_start = start

    best_window = audio[best_start:best_start + win_len]
    best_sec    = best_start / TARGET_SR
    print(f"    Best 3s window: {best_sec:.2f}s - {best_sec+win_sec:.2f}s  (RMS={best_rms:.5f})")
    return best_window, best_start

# ============================================================
# STEP 6: NORMALIZE
# ============================================================
def normalize(audio):
    mx = np.max(np.abs(audio))
    if mx > 0:
        return audio / mx
    return audio

# ============================================================
# STEP 7: PLOT & SAVE
# ============================================================
def plot_pipeline(raw_2k, filtered, gated, best_window, best_start, ts):
    out_png = os.path.join(OUTPUT_DIR, f"enhanced_{ts}.png")

    t_full = np.linspace(0, RECORD_SEC, len(raw_2k))
    t_clip = np.linspace(0, CLIP_SEC,   len(best_window))

    # RMS envelope for full signal
    def rms_env(audio, frame_sec=0.05):
        frame_len = int(TARGET_SR * frame_sec)
        hop       = frame_len // 2
        frames    = [audio[i:i+frame_len] for i in range(0, len(audio)-frame_len, hop)]
        return np.array([np.sqrt(np.mean(f**2)) for f in frames])

    env_raw      = rms_env(raw_2k)
    env_filtered = rms_env(filtered)
    env_gated    = rms_env(gated)
    env_best     = rms_env(best_window)
    t_env        = np.linspace(0, RECORD_SEC, len(env_raw))
    t_env_best   = np.linspace(0, CLIP_SEC,   len(env_best))

    # Beat detection on best window
    min_dist = int(0.35 / (0.025))   # ~0.35s min between beats (max 170 BPM)
    thresh   = np.percentile(env_best, 65)
    peaks, _ = sig.find_peaks(env_best, height=thresh, distance=min_dist)
    bpm = len(peaks) / CLIP_SEC * 60
    print(f"    Estimated BPM: {bpm:.0f}")

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f"Signal Enhancement Pipeline  |  {ts}", fontsize=13, fontweight='bold')

    # Col 1: Full 6s signal at each stage
    axes[0,0].plot(t_full, raw_2k, color='steelblue', lw=0.5)
    axes[0,0].set_title("1. Raw @ 2000 Hz  (full 6s)")
    axes[0,0].set_ylabel("Amplitude"); axes[0,0].grid(True, alpha=0.3)

    axes[1,0].plot(t_full, filtered, color='darkorange', lw=0.5)
    axes[1,0].set_title("2. After Bandpass + Notch Filter")
    axes[1,0].set_ylabel("Amplitude"); axes[1,0].grid(True, alpha=0.3)

    axes[2,0].plot(t_full, gated, color='purple', lw=0.5)
    axes[2,0].set_title(f"3. After Noise Gate (threshold={NOISE_GATE})")
    axes[2,0].set_ylabel("Amplitude"); axes[2,0].grid(True, alpha=0.3)

    # RMS envelope comparison (full signal)
    axes[3,0].plot(t_env, env_raw,      color='steelblue',  lw=1.0, label='raw',      alpha=0.7)
    axes[3,0].plot(t_env, env_filtered, color='darkorange',  lw=1.0, label='filtered', alpha=0.7)
    axes[3,0].plot(t_env, env_gated,    color='purple',      lw=1.0, label='gated',    alpha=0.7)
    # Mark selected window
    ws = best_start / TARGET_SR
    axes[3,0].axvspan(ws, ws + CLIP_SEC, alpha=0.2, color='green', label='selected 3s')
    axes[3,0].set_title("RMS Envelope - All Stages")
    axes[3,0].set_xlabel("Time (s)"); axes[3,0].legend(fontsize=7)
    axes[3,0].grid(True, alpha=0.3)

    # Col 2: Best 3s window detail
    axes[0,1].plot(t_clip, best_window, color='crimson', lw=0.8)
    axes[0,1].set_title(f"4. Best 3s Window (selected @ {best_start/TARGET_SR:.1f}s)")
    axes[0,1].set_ylabel("Amplitude"); axes[0,1].grid(True, alpha=0.3)

    axes[1,1].plot(t_env_best, env_best, color='crimson', lw=1.2)
    axes[1,1].plot(t_env_best[peaks], env_best[peaks], 'v',
                   color='navy', markersize=9, label=f'beats (~{bpm:.0f} BPM)')
    axes[1,1].axhline(thresh, color='gray', lw=1, linestyle=':', label='threshold')
    axes[1,1].set_title("Beat Detection on Best Window")
    axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.3)

    # FFT before vs after
    N      = len(best_window)
    freqs  = np.fft.rfftfreq(N, d=1.0/TARGET_SR)
    raw_seg = raw_2k[best_start:best_start + len(best_window)]
    if len(raw_seg) < N:
        raw_seg = np.pad(raw_seg, (0, N - len(raw_seg)))
    fft_raw  = 20*np.log10(np.abs(np.fft.rfft(raw_seg)) / N + 1e-9)
    fft_best = 20*np.log10(np.abs(np.fft.rfft(best_window)) / N + 1e-9)

    axes[2,1].plot(freqs, fft_raw,  color='steelblue',  lw=0.7, label='before', alpha=0.8)
    axes[2,1].plot(freqs, fft_best, color='crimson',     lw=0.7, label='after',  alpha=0.8)
    axes[2,1].set_title("Spectrum: Before vs After Enhancement")
    axes[2,1].set_xlabel("Frequency (Hz)"); axes[2,1].set_ylabel("dB")
    axes[2,1].set_xlim([0, 500])
    axes[2,1].axvspan(20, 150, alpha=0.1, color='green',  label='S1/S2')
    axes[2,1].axvspan(150, 400, alpha=0.07, color='yellow', label='murmur zone')
    axes[2,1].legend(fontsize=7); axes[2,1].grid(True, alpha=0.3)

    # Mel Spectrogram of best window
    try:
        import librosa
        mel    = librosa.feature.melspectrogram(y=best_window, sr=TARGET_SR,
                    n_mels=64, n_fft=512, hop_length=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = axes[3,1].imshow(mel_db, aspect='auto', origin='lower',
                               cmap='inferno',
                               extent=[0, CLIP_SEC, 0, 64])
        axes[3,1].set_title("Mel Spectrogram (model input)")
        axes[3,1].set_xlabel("Time (s)"); axes[3,1].set_ylabel("Mel band")
        plt.colorbar(img, ax=axes[3,1], label='dB')
    except ImportError:
        axes[3,1].text(0.5, 0.5, 'librosa not available', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {out_png}")
    return out_png

# ============================================================
# MAIN
# ============================================================
def main():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pipeline
    raw_44k  = record()
    raw_2k   = downsample(raw_44k)

    print("\n[2] Applying filters...")
    filtered = apply_filters(raw_2k)

    print("\n[3] Applying noise gate...")
    gated, mask = noise_gate(filtered)

    print("\n[4] Finding best 3s window...")
    best_window, best_start = find_best_window(gated)

    print("\n[5] Normalizing...")
    best_norm = normalize(best_window)

    print("\n[6] Saving...")
    # Save WAVs
    raw_path  = os.path.join(OUTPUT_DIR, f"enhanced_{ts}_raw3s.wav")
    best_path = os.path.join(OUTPUT_DIR, f"enhanced_{ts}_best3s.wav")

    raw_seg = normalize(raw_2k[best_start:best_start + SAMPLES])
    wav.write(raw_path,  TARGET_SR, (raw_seg  * 32767).astype(np.int16))
    wav.write(best_path, TARGET_SR, (best_norm * 32767).astype(np.int16))
    print(f"    Raw 3s  WAV: {raw_path}")
    print(f"    Best 3s WAV: {best_path}")

    print("\n[7] Plotting pipeline...")
    plot_pipeline(raw_2k, filtered, gated, best_norm, best_start, ts)

    print("\n=== Summary ===")
    print(f"  Noise gate threshold : {NOISE_GATE} (adjust if needed)")
    print(f"  Raw 3s RMS           : {np.sqrt(np.mean(raw_seg**2)):.5f}")
    print(f"  Enhanced 3s RMS      : {np.sqrt(np.mean(best_norm**2)):.5f}")
    print(f"  Files saved to       : {OUTPUT_DIR}")
    print("  Next: feed enhanced_{ts}_best3s.wav to model and compare")

if __name__ == "__main__":
    main()
