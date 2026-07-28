"""
Unit tests for the DSP pipeline. These run with plain numpy/scipy/librosa
-- no PySide6, no microphone, no trained model required -- which is the
whole point of having pulled this logic out of the GUI file.
"""
import numpy as np
import pytest

from smart_stethoscope.config import Settings
from smart_stethoscope.signal_pipeline import (
    bandpass_filter,
    enhance_pipeline,
    estimate_bpm,
    extract_mel,
    find_best_window,
    noise_gate,
    notch_filter,
)


@pytest.fixture
def settings():
    return Settings()


def _synthetic_heartbeat(sample_rate=2000, duration=8, bpm=72, noise=0.05, seed=0):
    """A crude but periodic 'lub-dub'-like signal: two damped pulses per
    beat, so peak-detection has something regular to lock onto."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    beat_period = 60.0 / bpm
    signal = np.zeros_like(t)
    beat_times = np.arange(0, duration, beat_period)
    for bt in beat_times:
        signal += np.exp(-((t - bt) ** 2) / (2 * 0.01 ** 2))
        signal += 0.6 * np.exp(-((t - bt - 0.15) ** 2) / (2 * 0.01 ** 2))
    signal += rng.normal(0, noise, size=t.shape)
    return signal.astype(np.float32)


def test_bandpass_filter_preserves_length(settings):
    audio = _synthetic_heartbeat(sample_rate=settings.sample_rate)
    out = bandpass_filter(audio, settings.sample_rate)
    assert out.shape == audio.shape
    assert np.all(np.isfinite(out))


def test_bandpass_filter_degrades_gracefully_on_bad_input(settings):
    # A single-sample "signal" can't support filtfilt's padding -- the
    # function must return the input unchanged rather than raise.
    out = bandpass_filter(np.array([1.0], dtype=np.float32), settings.sample_rate)
    assert out.shape == (1,)


def test_notch_filter_preserves_length(settings):
    audio = _synthetic_heartbeat(sample_rate=settings.sample_rate)
    out = notch_filter(audio, settings.sample_rate)
    assert out.shape == audio.shape


def test_noise_gate_zeros_silent_regions(settings):
    audio = np.zeros(settings.sample_rate * 3, dtype=np.float32)
    # One loud burst in the middle, silence everywhere else.
    audio[1000:1100] = 1.0
    gated = noise_gate(audio, settings.sample_rate, gate_ratio=0.5)
    assert gated[0] == 0.0
    assert gated[-1] == 0.0


def test_find_best_window_returns_requested_length(settings):
    audio = _synthetic_heartbeat(sample_rate=settings.sample_rate, duration=8)
    window, start = find_best_window(audio, settings.sample_rate, settings.duration)
    assert len(window) == settings.sample_rate * settings.duration
    assert 0 <= start <= len(audio) - len(window)


def test_estimate_bpm_is_in_plausible_range(settings):
    audio = _synthetic_heartbeat(sample_rate=settings.sample_rate, duration=8, bpm=72)
    bpm = estimate_bpm(audio, settings.sample_rate, settings.duration)
    assert 30 <= bpm <= 200


def test_extract_mel_matches_model_input_shape(settings):
    audio = _synthetic_heartbeat(sample_rate=settings.sample_rate, duration=settings.duration)
    mel = extract_mel(audio, settings)
    # (n_mels, n_frames, 1) -- batch dimension added later by inference.py
    assert mel.shape[0] == settings.n_mels
    assert mel.shape[2] == 1
    assert np.all(np.isfinite(mel))


def test_enhance_pipeline_normalizes_to_unit_range(settings):
    raw = _synthetic_heartbeat(sample_rate=settings.capture_sr, duration=settings.record_sec, noise=0.02)
    best, raw_ds, filtered, gated, best_start = enhance_pipeline(raw, settings)
    assert len(best) == settings.sample_rate * settings.duration
    assert np.max(np.abs(best)) <= 1.0 + 1e-6
    assert best_start >= 0


def test_enhance_pipeline_handles_silence_without_crashing(settings):
    raw = np.zeros(settings.capture_sr * settings.record_sec, dtype=np.float32)
    best, *_rest = enhance_pipeline(raw, settings)
    assert len(best) == settings.sample_rate * settings.duration
    assert np.all(best == 0.0)
