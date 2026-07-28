"""
Audio signal-processing pipeline for heart sound recordings.

Extracted from the original monolithic GUI script (stethoscope_gui_v7.py)
into pure, framework-independent functions with no PySide6/Qt dependency.
This is what makes the pipeline unit-testable and reusable outside the GUI
(e.g. from a Jupyter notebook, a CLI batch-scorer, or a future web API).

Pipeline (matches what the model was trained on):
    Capture @ 44100 Hz
    -> Downsample to 2000 Hz
    -> Bandpass (20-950 Hz) + Notch (50 Hz)
    -> Noise gate (drop low-energy segments)
    -> Best-window selection (most regular heartbeat pattern in a
       `duration`-second slice, not just highest RMS -- avoids picking a
       cough/friction artifact as "the loudest part")
    -> Normalize to [-1, 1]
    -> Mel-spectrogram feature extraction (must match training config)
"""
from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, decimate, filtfilt, find_peaks, iirnotch

from .config import Settings


def bandpass_filter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """4th-order Butterworth bandpass, 20 Hz - 950 Hz (clamped to Nyquist)."""
    try:
        nyq = sample_rate / 2.0
        high = min(0.95, 950.0 / nyq)
        b, a = butter(4, [20.0 / nyq, high], btype="band")
        return filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)
    except Exception:
        # A failed filter should degrade gracefully (unfiltered audio),
        # not crash a live diagnostic session.
        return audio


def notch_filter(audio: np.ndarray, sample_rate: int, freq_hz: float = 50.0) -> np.ndarray:
    """Notch out mains hum (50 Hz default; use 60.0 for North American mains)."""
    try:
        nyq = sample_rate / 2.0
        b, a = iirnotch(freq_hz / nyq, 30)
        return filtfilt(b, a, audio.astype(np.float64)).astype(np.float32)
    except Exception:
        return audio


def noise_gate(audio: np.ndarray, sample_rate: int, gate_ratio: float, frame_sec: float = 0.05) -> np.ndarray:
    """Zero out frames whose RMS energy falls below `gate_ratio` * peak RMS."""
    frame_len = int(sample_rate * frame_sec)
    hop = frame_len // 2
    output = audio.copy()
    rms_vals, positions = [], []
    for i in range(0, len(audio) - frame_len, hop):
        rms_vals.append(np.sqrt(np.mean(audio[i:i + frame_len] ** 2)))
        positions.append(i)
    if not rms_vals:
        return output
    rms_vals = np.array(rms_vals)
    threshold = gate_ratio * np.max(rms_vals)
    mask = np.zeros(len(audio), dtype=bool)
    for idx, i in enumerate(positions):
        if rms_vals[idx] >= threshold:
            mask[i:min(i + frame_len, len(audio))] = True
    output[~mask] = 0.0
    return output


def find_best_window(audio: np.ndarray, sample_rate: int, win_sec: int):
    """
    Select the `win_sec`-second window with the most REGULAR heartbeat
    pattern, using peak-interval regularity rather than raw RMS -- a loud
    cough or stethoscope-friction burst should not win over a quieter but
    genuinely periodic heart sound.
    """
    win_len = int(sample_rate * win_sec)
    hop = int(sample_rate * 0.1)
    best_score = -1.0
    best_start = 0
    frame_len = int(sample_rate * 0.05)
    hop2 = frame_len // 2

    for start in range(0, max(1, len(audio) - win_len), hop):
        window = audio[start:start + win_len]

        env_frames = [window[i:i + frame_len] for i in range(0, len(window) - frame_len, hop2)]
        if len(env_frames) < 4:
            continue
        env = np.array([np.sqrt(np.mean(f ** 2)) for f in env_frames])

        min_dist = max(1, int(0.35 / (hop2 / sample_rate)))
        thresh = np.percentile(env, 55)
        peaks, _ = find_peaks(env, height=thresh, distance=min_dist)
        n_peaks = len(peaks)
        if n_peaks < 2:
            continue

        intervals = np.diff(peaks)
        regularity = 1.0 / (1.0 + np.std(intervals) / (np.mean(intervals) + 1e-6))

        rms = np.sqrt(np.mean(window ** 2))
        global_rms = np.sqrt(np.mean(audio ** 2))
        rms_penalty = max(0.0, rms / (global_rms + 1e-9) - 3.0)

        score = n_peaks * regularity / (1.0 + rms_penalty)
        if score > best_score:
            best_score = score
            best_start = start

    return audio[best_start:best_start + win_len], best_start


def estimate_bpm(audio: np.ndarray, sample_rate: int, duration: int) -> int:
    frame_len = int(sample_rate * 0.05)
    hop = frame_len // 2
    frames = [audio[i:i + frame_len] for i in range(0, len(audio) - frame_len, hop)]
    if not frames:
        return 0
    env = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    min_dist = int(0.35 / (hop / sample_rate))
    thresh = np.percentile(env, 60)
    peaks, _ = find_peaks(env, height=thresh, distance=min_dist)
    bpm = len(peaks) / duration * 60
    return int(np.clip(bpm, 30, 200))


def extract_mel(audio: np.ndarray, settings: Settings) -> np.ndarray:
    """Mel-spectrogram in dB, shape (n_mels, n_frames, 1) -- exactly the
    model's expected input tensor shape (minus batch dim)."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=settings.sample_rate,
        n_mels=settings.n_mels, n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db[..., np.newaxis].astype(np.float32)


def enhance_pipeline(raw_audio: np.ndarray, settings: Settings):
    """
    Full pipeline from a raw capture (at settings.capture_sr) down to a
    normalized, best-window `duration`-second clip ready for `extract_mel`.

    Returns (best_window, raw_downsampled, filtered, gated, best_start_sample)
    -- the intermediate stages are kept because the GUI/report code plots
    all of them for clinical transparency (what did the model actually see).
    """
    factor = settings.capture_sr // settings.sample_rate
    raw_2k = decimate(raw_audio, factor, ftype="fir", zero_phase=True).astype(np.float32)
    filtered = bandpass_filter(raw_2k, settings.sample_rate)
    filtered = notch_filter(filtered, settings.sample_rate)
    gated = noise_gate(filtered, settings.sample_rate, settings.noise_gate)
    best, best_start = find_best_window(gated, settings.sample_rate, settings.duration)
    mx = np.max(np.abs(best))
    if mx > 0:
        best = best / mx
    return best.astype(np.float32), raw_2k, filtered, gated, best_start
