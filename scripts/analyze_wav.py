# NOTE: standalone hardware bring-up / R&D script, kept for reference.
# It predates the src/smart_stethoscope/ package and still uses the
# original hardcoded paths (~/Desktop/Smart_Stethoscope) -- adjust the
# SETTINGS/paths at the top before running it on your own machine.
#!/usr/bin/env python3
"""
Analyze recorded WAV - check for clipping and heart sound quality
Smart Stethoscope Project
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import glob

# ============================================================
# Find latest WAV automatically
# ============================================================
OUTPUT_DIR = os.path.expanduser("~/Desktop/Smart_Stethoscope")

wavfiles = sorted(glob.glob(os.path.join(OUTPUT_DIR, "heart_raw_*.wav")))
if not wavfiles:
    print("No heart_raw_*.wav found in", OUTPUT_DIR)
    sys.exit(1)

wav_path = wavfiles[-1]   # latest
print(f"Analyzing: {wav_path}\n")

# ============================================================
sr, data = wav.read(wav_path)
audio = data.astype(np.float32) / 32767.0   # normalize to -1..1

print(f"Sample rate : {sr} Hz")
print(f"Duration    : {len(audio)/sr:.1f} s")
print(f"Max |amp|   : {np.max(np.abs(audio)):.4f}")

# ============================================================
# Clipping detection
# ============================================================
CLIP_THRESH = 0.98
clipped_samples = np.sum(np.abs(audio) >= CLIP_THRESH)
clip_pct = 100.0 * clipped_samples / len(audio)
print(f"\nClipping ({CLIP_THRESH}) : {clipped_samples} samples ({clip_pct:.2f}%)")
if clip_pct > 1.0:
    print("  [!!] HEAVY CLIPPING - run: amixer -c 3 sset Mic 60%")
elif clip_pct > 0.1:
    print("  [!] Mild clipping   - run: amixer -c 3 sset Mic 75%")
else:
    print("  [OK] No significant clipping")

# ============================================================
# Downsample + filter (identical to training pipeline)
# ============================================================
TARGET_SR = 2000
factor = sr // TARGET_SR
audio_2k = signal.decimate(audio, factor, ftype='fir', zero_phase=True)

nyq  = TARGET_SR / 2.0
low  = 20.0 / nyq
high = min(0.95, 950.0 / nyq)
b, a = signal.butter(4, [low, high], btype='band')
filtered = signal.filtfilt(b, a, audio_2k)

b_n, a_n = signal.iirnotch(50.0, 30, TARGET_SR)
filtered  = signal.filtfilt(b_n, a_n, filtered)

rms  = np.sqrt(np.mean(filtered**2))
peak = np.max(np.abs(filtered))
print(f"\nAfter pipeline (2000 Hz + bandpass + notch):")
print(f"  RMS  : {rms:.6f}")
print(f"  Peak : {peak:.6f}")

# ============================================================
# Heartbeat detection (simple peak finding on RMS envelope)
# ============================================================
frame_len = int(TARGET_SR * 0.05)
hop       = frame_len // 2
frames    = [filtered[i:i+frame_len] for i in range(0, len(filtered)-frame_len, hop)]
rms_env   = np.array([np.sqrt(np.mean(f**2)) for f in frames])
t_env     = np.linspace(0, len(audio)/sr, len(rms_env))

# find peaks with minimum distance ~0.4s (max 150 BPM)
min_dist  = int(0.4 / (hop / TARGET_SR))
thresh    = np.percentile(rms_env, 70)
peaks, _  = signal.find_peaks(rms_env, height=thresh, distance=min_dist)
bpm       = len(peaks) / (len(audio)/sr) * 60.0
print(f"\nEstimated heart rate : {bpm:.0f} BPM  ({len(peaks)} beats in {len(audio)/sr:.0f}s)")
if 40 <= bpm <= 180:
    print("  [OK] Plausible heart rate detected!")
else:
    print("  [?] Unusual rate - may be noise, check plot")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 11))
fig.suptitle(f"Heart Sound Analysis | {os.path.basename(wav_path)}", fontsize=12, fontweight='bold')

t_raw  = np.linspace(0, len(audio)/sr, len(audio))
t_filt = np.linspace(0, len(audio)/sr, len(filtered))

# Raw waveform with clipping highlight
axes[0].plot(t_raw, audio, color='steelblue', linewidth=0.4, label='raw')
axes[0].axhline( CLIP_THRESH, color='red', lw=1.2, linestyle='--', label=f'clip threshold ({CLIP_THRESH})')
axes[0].axhline(-CLIP_THRESH, color='red', lw=1.2, linestyle='--')
axes[0].fill_between(t_raw, CLIP_THRESH, 1.0,  color='red', alpha=0.15)
axes[0].fill_between(t_raw, -1.0, -CLIP_THRESH, color='red', alpha=0.15)
axes[0].set_title(f"Raw Waveform @ {sr} Hz  |  Clipping: {clip_pct:.2f}%")
axes[0].set_ylabel("Amplitude")
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Filtered waveform
axes[1].plot(t_filt, filtered, color='darkorange', linewidth=0.7)
axes[1].set_title("Filtered Signal @ 2000 Hz  (Bandpass 20-950 Hz + Notch 50 Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True, alpha=0.3)

# Frequency spectrum
N     = len(filtered)
freqs = np.fft.rfftfreq(N, d=1.0/TARGET_SR)
fft_m = np.abs(np.fft.rfft(filtered)) / N
axes[2].plot(freqs, 20*np.log10(fft_m + 1e-9), color='green', linewidth=0.7)
axes[2].axvspan(20,  150, alpha=0.12, color='green',  label='S1/S2 zone (20-150 Hz)')
axes[2].axvspan(150, 400, alpha=0.08, color='yellow', label='murmur zone (150-400 Hz)')
axes[2].set_title("Frequency Spectrum (dB)")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Magnitude (dB)")
axes[2].set_xlim([0, 500])
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

# RMS envelope + detected beats
axes[3].plot(t_env, rms_env, color='crimson', linewidth=1.0, label='RMS energy')
axes[3].plot(t_env[peaks], rms_env[peaks], 'v', color='navy', markersize=8, label=f'beats (~{bpm:.0f} BPM)')
axes[3].axhline(thresh, color='gray', lw=1, linestyle=':', label='detection threshold')
axes[3].set_title("RMS Energy + Heartbeat Detection")
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("RMS")
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
out_png = wav_path.replace('.wav', '_analysis.png')
plt.savefig(out_png, dpi=120, bbox_inches='tight')
plt.close()
print(f"\nPlot saved: {out_png}")
