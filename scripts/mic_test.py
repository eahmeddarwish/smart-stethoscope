# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
"""
Heart Sound Microphone Diagnostic
Smart Stethoscope Project - Group 07 KCST
Tests microphone capture and shows signal quality
"""

import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('Agg')  # No display needed on Pi - saves PNG
import matplotlib.pyplot as plt
import scipy.signal as signal
import datetime
import os

# ============================================================
# SETTINGS - adjust if needed
# ============================================================
DURATION     = 10       # seconds to record
SAMPLE_RATE  = 44100    # high rate for diagnosis (we'll downsample to 2000 later)
CHANNELS     = 1
OUTPUT_DIR   = os.path.expanduser("~/Desktop/Smart_Stethoscope")

# ============================================================

def list_audio_devices():
    print("\n=== Audio Devices ===")
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"  [{i}] INPUT:  {d['name']}  (ch={d['max_input_channels']}, rate={int(d['default_samplerate'])})")
    print("====================\n")

def record_audio(device=None):
    print(f"Recording {DURATION}s at {SAMPLE_RATE} Hz...")
    print(">>> ضع السماعة على صدرك والان <<<")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32',
        device=device
    )
    sd.wait()
    audio = audio.flatten()
    print(f"Done. Samples: {len(audio)}, Max amplitude: {np.max(np.abs(audio)):.4f}")
    return audio

def analyze_and_plot(audio):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"heart_test_{timestamp}.png")

    # --- Downsample to 2000 Hz (same as model pipeline) ---
    target_sr = 2000
    downsample_factor = SAMPLE_RATE // target_sr
    audio_2k = signal.decimate(audio, downsample_factor, ftype='fir', zero_phase=True)

    # --- Bandpass filter 20-950 Hz (same as training) ---
    nyq = target_sr / 2.0
    low  = 20.0  / nyq
    high = min(0.95, 950.0 / nyq)
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio_2k)

    # --- Notch filter 50 Hz ---
    b_notch, a_notch = signal.iirnotch(50.0, 30, target_sr)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)

    # --- RMS energy (proxy for signal quality) ---
    rms = np.sqrt(np.mean(filtered**2))
    peak = np.max(np.abs(filtered))
    snr_estimate = 20 * np.log10(peak / (rms + 1e-9))

    print("\n=== Signal Quality Report ===")
    print(f"  RMS amplitude  : {rms:.6f}")
    print(f"  Peak amplitude : {peak:.6f}")
    print(f"  SNR estimate   : {snr_estimate:.1f} dB")
    if rms < 0.001:
        print("  STATUS: [!!] Signal too weak - increase amixer gain")
    elif rms < 0.01:
        print("  STATUS: [OK but low] - consider slight gain increase")
    else:
        print("  STATUS: [GOOD] - signal looks usable")
    print("=============================\n")

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f"Heart Sound Diagnostic  |  {timestamp}", fontsize=13, fontweight='bold')

    t_raw = np.linspace(0, DURATION, len(audio))
    t_filt = np.linspace(0, DURATION, len(filtered))

    # 1. Raw waveform (full rate)
    axes[0].plot(t_raw, audio, color='steelblue', linewidth=0.5)
    axes[0].set_title("Raw Signal @ 44100 Hz")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.3)

    # 2. Filtered waveform @ 2000 Hz
    axes[1].plot(t_filt, filtered, color='darkorange', linewidth=0.8)
    axes[1].set_title("Filtered Signal @ 2000 Hz  (Bandpass 20-950 Hz + Notch 50 Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    # 3. FFT spectrum of filtered signal
    N = len(filtered)
    freqs = np.fft.rfftfreq(N, d=1.0/target_sr)
    fft_mag = np.abs(np.fft.rfft(filtered)) / N
    axes[2].plot(freqs, 20*np.log10(fft_mag + 1e-9), color='green', linewidth=0.7)
    axes[2].set_title("Frequency Spectrum (dB)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude (dB)")
    axes[2].set_xlim([0, 500])   # Heart sounds are mostly 20-400 Hz
    axes[2].axvspan(20, 400, alpha=0.1, color='green', label='Heart sound band')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # 4. RMS energy over time (detects heartbeat peaks)
    frame_len = int(target_sr * 0.05)   # 50ms frames
    hop = frame_len // 2
    frames = [filtered[i:i+frame_len] for i in range(0, len(filtered)-frame_len, hop)]
    rms_curve = np.array([np.sqrt(np.mean(f**2)) for f in frames])
    t_rms = np.linspace(0, DURATION, len(rms_curve))
    axes[3].plot(t_rms, rms_curve, color='crimson', linewidth=1.2)
    axes[3].set_title("RMS Energy Over Time  (peaks = heartbeats)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("RMS")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")
    return out_path

def main():
    list_audio_devices()

    # Auto-select device or ask user
    device_input = input("Enter device number (or press Enter for default): ").strip()
    device = int(device_input) if device_input else None

    audio = record_audio(device=device)
    analyze_and_plot(audio)

    # Also save raw WAV for re-analysis later
    import scipy.io.wavfile as wav
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(OUTPUT_DIR, f"heart_raw_{ts}.wav")
    wav.write(wav_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Raw WAV saved: {wav_path}")

if __name__ == "__main__":
    main()