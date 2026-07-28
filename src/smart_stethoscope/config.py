"""
Central configuration for Smart Stethoscope.

Everything that used to be hardcoded to a single Raspberry Pi user's home
directory (`/home/pi/Desktop/Smart_Stethoscope`) now lives here and can be
overridden with environment variables. This is the single biggest change
needed to let anyone else build on this project without editing source code.

Environment variables (all optional, sensible defaults provided):
    STETHOSCOPE_BASE_DIR    Root folder for models/reports/datasets
    STETHOSCOPE_MODEL_NAME  Filename of the .tflite model inside <base>/models
    STETHOSCOPE_ENCODER_NAME
    STETHOSCOPE_CONFIG_NAME
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field


def _default_base_dir() -> str:
    # Falls back to a folder next to the user's home directory on any OS,
    # instead of assuming a Raspberry Pi desktop user account exists.
    return os.environ.get(
        "STETHOSCOPE_BASE_DIR",
        os.path.join(os.path.expanduser("~"), "Smart_Stethoscope"),
    )


@dataclass
class Settings:
    base_dir: str = field(default_factory=_default_base_dir)
    model_name: str = os.environ.get("STETHOSCOPE_MODEL_NAME", "heart_model_v9_int8.tflite")
    encoder_name: str = os.environ.get("STETHOSCOPE_ENCODER_NAME", "heart_label_encoder_v9.pkl")
    config_name: str = os.environ.get("STETHOSCOPE_CONFIG_NAME", "heart_config_v9.pkl")

    # Audio pipeline defaults (must match training exactly; can be
    # overridden by a heart_config_v9.pkl shipped next to the model).
    sample_rate: int = 2000
    duration: int = 3
    n_mels: int = 64
    n_fft: int = 512
    hop_length: int = 128
    murmur_threshold: float = 0.50
    murmur_idx: int = 2
    capture_sr: int = 44100
    record_sec: int = 8
    noise_gate: float = 0.30

    @property
    def model_path(self) -> str:
        return os.path.join(self.base_dir, "models", self.model_name)

    @property
    def encoder_path(self) -> str:
        return os.path.join(self.base_dir, "models", self.encoder_name)

    @property
    def config_path(self) -> str:
        return os.path.join(self.base_dir, "models", self.config_name)

    @property
    def reports_dir(self) -> str:
        d = os.path.join(self.base_dir, "reports")
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def dataset_dir(self) -> str:
        return os.path.join(self.base_dir, "datasets", "Heartbeats")

    def load_overrides_from_pickle(self) -> None:
        """If a heart_config_v9.pkl ships with the model, trust it over the
        hardcoded defaults above -- it reflects exactly what the model was
        trained with."""
        if not os.path.exists(self.config_path):
            return
        with open(self.config_path, "rb") as f:
            cfg = pickle.load(f)
        self.sample_rate = cfg.get("sample_rate", self.sample_rate)
        self.n_mels = cfg.get("n_mels", self.n_mels)
        self.n_fft = cfg.get("n_fft", self.n_fft)
        self.hop_length = cfg.get("hop_length", self.hop_length)
        self.duration = cfg.get("duration", self.duration)
        self.murmur_threshold = cfg.get("murmur_threshold", self.murmur_threshold)
        self.murmur_idx = cfg.get("murmur_idx", self.murmur_idx)

    @property
    def samples(self) -> int:
        return self.sample_rate * self.duration


SETTINGS = Settings()
SETTINGS.load_overrides_from_pickle()
