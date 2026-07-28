# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
"""
Heart Sound Filter Analysis
Records audio then shows before/after filtering comparison
Smart Stethoscope Project - Group 07 KCST
"""

import numpy as np
import sounddevice as sd
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import os

# ============================================================
# SETTINGS
# ============================================================
DURATION    = 10
SAMPLE_RATE = 44100
TARGET_SR   = 2000
DEVICE      = 1
OUTPUT_DIR  = os.path.expanduser("~/Desktop/Smart_Stethoscope")

# ============================================================

def record():
    print(f"Recording {DURATION}s ... ضع السماعة على صدرك")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32', device=DEVICE)
    sd.wait()
    audio = audio.flatten()
    print(f"Done. Peak={np.max(np.abs(audio)):.4f}")
    return audio

def downsample(audio):
    factor = SAMPLE_RATE // TARGET_SR
    return signal.decimate(audio, factor, ftype='fir', zero_phase=True)

def apply_filters(audio_2k):
    nyq = TARGET_SR / 2.0

    # Bandpass 20-950 Hz
    low  = 20.0 / nyq
    high = min(0.95, 950.0 / nyq)
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio_2k)

    # Notch 50 Hz
    b_n, a_n = signal.iirnotch(50.0, 30, TARGET_SR)
    filtered  = signal.filtfilt(b_n, a_n, filtered)

    return filtered

def rms_envelope(audio, frame_sec=0.05):
    frame_len = int(TARGET_SR * frame_sec)
    hop       = frame_len // 2
    frames    = [audio[i:i+frame_len] for i in range(0, len(audio)-frame_len, hop)]
    return np.array([np.sqrt(np.mean(f**2)) for f in frames])

def analyze_and_plot(raw_44k, raw_2k, filtered_2k):
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(OUTPUT_DIR, f"filter_compare_{ts}.png")

    t_raw  = np.linspace(0, DURATION, len(raw_2k))
    t_filt = np.linspace(0, DURATION, len(filtered_2k))

    env_before = rms_envelope(raw_2k)
    env_after  = rms_envelope(filtered_2k)
    t_env      = np.linspace(0, DURATION, len(env_before))

    # SNR comparison
    rms_before = np.sqrt(np.mean(raw_2k**2))
    rms_after  = np.sqrt(np.mean(filtered_2k**2))
    peak_before = np.max(np.abs(raw_2k))
    peak_after  = np.max(np.abs(filtered_2k))

    print("\n=== Filter Comparison Report ===")
    print(f"  Before | RMS={rms_before:.6f}  Peak={peak_before:.6f}")
    print(f"  After  | RMS={rms_after:.6f}  Peak={peak_after:.6f}")
    print(f"  Signal reduction: {20*np.log10(rms_after/(rms_before+1e-9)):.1f} dB")

    # Clipping check
    clip_pct = 100.0 * np.sum(np.abs(raw_2k) >= 0.98) / len(raw_2k)
    print(f"  Clipping: {clip_pct:.2f}%")
    if clip_pct < 0.1:
        print("  [OK] No clipping")
    else:
        print(f"  [!!] Clipping detected - lower gain")
    print("================================\n")

    # ---- FFT ----
    N      = len(raw_2k)
    freqs  = np.fft.rfftfreq(N, d=1.0/TARGET_SR)
    fft_b  = 20*np.log10(np.abs(np.fft.rfft(raw_2k)) / N + 1e-9)
    fft_a  = 20*np.log10(np.abs(np.fft.rfft(filtered_2k)) / N + 1e-9)

    # ---- Plot (6 subplots) ----
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Filter Comparison  |  {ts}", fontsize=13, fontweight='bold')

    # Row 1: Waveforms
    axes[0,0].plot(t_raw, raw_2k, color='steelblue', linewidth=0.6)
    axes[0,0].set_title("BEFORE Filter @ 2000 Hz")
    axes[0,0].set_ylabel("Amplitude")
    axes[0,0].axhline( 0.98, color='red', lw=1, linestyle='--', alpha=0.5)
    axes[0,0].axhline(-0.98, color='red', lw=1, linestyle='--', alpha=0.5)
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(t_filt, filtered_2k, color='darkorange', linewidth=0.6)
    axes[0,1].set_title("AFTER Filter (Bandpass 20-950 Hz + Notch 50 Hz)")
    axes[0,1].set_ylabel("Amplitude")
    axes[0,1].grid(True, alpha=0.3)

    # Row 2: FFT
    axes[1,0].plot(freqs, fft_b, color='steelblue', linewidth=0.7)
    axes[1,0].set_title("Spectrum BEFORE (dB)")
    axes[1,0].set_xlabel("Frequency (Hz)")
    axes[1,0].set_ylabel("Magnitude (dB)")
    axes[1,0].set_xlim([0, 500])
    axes[1,0].axvspan(20, 150, alpha=0.12, color='green', label='S1/S2')
    axes[1,0].axvspan(150, 400, alpha=0.08, color='yellow', label='murmur')
    axes[1,0].axvline(50, color='red', lw=1, linestyle=':', label='50Hz noise')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(freqs, fft_a, color='darkorange', linewidth=0.7)
    axes[1,1].set_title("Spectrum AFTER (dB)")
    axes[1,1].set_xlabel("Frequency (Hz)")
    axes[1,1].set_ylabel("Magnitude (dB)")
    axes[1,1].set_xlim([0, 500])
    axes[1,1].axvspan(20, 150, alpha=0.12, color='green', label='S1/S2')
    axes[1,1].axvspan(150, 400, alpha=0.08, color='yellow', label='murmur')
    axes[1,1].axvline(50, color='red', lw=1, linestyle=':', label='50Hz notched')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    # Row 3: RMS Envelope comparison
    axes[2,0].plot(t_env, env_before, color='steelblue', linewidth=1.0)
    axes[2,0].set_title("RMS Envelope BEFORE (heartbeat peaks)")
    axes[2,0].set_xlabel("Time (s)")
    axes[2,0].set_ylabel("RMS")
    axes[2,0].grid(True, alpha=0.3)

    axes[2,1].plot(t_env, env_after, color='crimson', linewidth=1.0)
    axes[2,1].set_title("RMS Envelope AFTER (heartbeat peaks)")
    axes[2,1].set_xlabel("Time (s)")
    axes[2,1].set_ylabel("RMS")
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_png}")

    # Save WAV before/after
    wav.write(out_png.replace('.png','_before.wav'), TARGET_SR,
              (raw_2k * 32767).astype(np.int16))
    wav.write(out_png.replace('.png','_after.wav'), TARGET_SR,
              (filtered_2k * 32767).astype(np.int16))
    print("WAV files saved (before + after)")

def main():
    # Make sure gain is set
    os.system("amixer -c 3 sset Mic 30% > /dev/null 2>&1")
    print("Gain set to 30%")

    raw_44k    = record()
    raw_2k     = downsample(raw_44k)
    filtered_2k = apply_filters(raw_2k)
    analyze_and_plot(raw_44k, raw_2k, filtered_2k)

if __name__ == "__main__":
    main()
