"""
Audio input: either a real microphone recording, or (demo mode) a random
WAV file drawn from the training dataset. Kept separate from the GUI so
it can be exercised in a script or notebook without starting PySide6.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from .config import Settings
from .signal_pipeline import enhance_pipeline

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


@dataclass
class CapturedAudio:
    audio: np.ndarray       # best-window, normalized, ready for extract_mel
    raw_2k: np.ndarray      # full downsampled capture (for plotting)
    filtered: np.ndarray
    gated: np.ndarray
    best_start: int
    source: str             # '' for a live mic capture, else the WAV path used


def pick_random_demo_wav(settings: Settings) -> Optional[str]:
    """Used only in demo/simulation mode -- picks a random recording from
    the local dataset copy so the GUI can be shown/tested without a real
    stethoscope attached. Never used for anything resembling a real
    diagnosis: `source` is always populated with the file path so the
    report clearly says which archived recording was used."""
    patterns = [
        os.path.join(settings.dataset_dir, "set_a", "*.wav"),
        os.path.join(settings.dataset_dir, "set_b", "*.wav"),
        os.path.join(settings.base_dir, "*.wav"),
    ]
    wavs: list[str] = []
    for p in patterns:
        wavs.extend(glob.glob(p))
    return random.choice(wavs) if wavs else None


def load_demo_wav(path: str, settings: Settings) -> CapturedAudio:
    audio_raw, _ = librosa.load(path, sr=settings.capture_sr)
    audio, raw_2k, filtered, gated, best_start = enhance_pipeline(audio_raw, settings)
    return CapturedAudio(audio, raw_2k, filtered, gated, best_start, source=path)


def record_from_microphone(settings: Settings) -> CapturedAudio:
    """Blocking call -- run this on a worker thread, not the Qt UI thread."""
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError(
            "sounddevice is not installed/available. Install it with "
            "'pip install sounddevice' and make sure PortAudio is present "
            "on the system (e.g. `sudo apt install libportaudio2` on Raspberry Pi OS)."
        )
    n_frames = int(settings.capture_sr * settings.record_sec)
    raw = sd.rec(n_frames, samplerate=settings.capture_sr, channels=1, dtype="float32", device=None)
    sd.wait()
    raw = raw.flatten()
    audio, raw_2k, filtered, gated, best_start = enhance_pipeline(raw, settings)
    return CapturedAudio(audio, raw_2k, filtered, gated, best_start, source="")
