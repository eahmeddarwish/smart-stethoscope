"""
Model loading and inference, decoupled from Qt.

The interpreter fallback chain below matters: on a Raspberry Pi, the
lightest available runtime should always win, since installing full
TensorFlow on a Pi is slow and heavier than necessary just to run an
already-quantized .tflite model.

Order of preference:
    1. ai_edge_litert   (Google's actively-maintained successor to
                          tflite_runtime; what the project was built against)
    2. tflite_runtime    (older standalone package, smaller than full TF)
    3. tensorflow.lite   (last resort -- works everywhere but heaviest)
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from .clinical_info import format_reasoning
from .config import Settings

try:
    from ai_edge_litert.interpreter import Interpreter  # type: ignore
except ImportError:
    try:
        import tflite_runtime.interpreter as _tfl  # type: ignore

        Interpreter = _tfl.Interpreter
    except ImportError:
        import tensorflow as tf  # type: ignore

        Interpreter = tf.lite.Interpreter


@dataclass
class InferenceResult:
    pred: str
    probs: List[float]
    classes: List[str]
    murmur_p: float
    reasoning: str


class HeartSoundClassifier:
    """Thin wrapper around a loaded TFLite interpreter + label encoder."""

    def __init__(self, settings: Settings):
        self.settings = settings
        with open(settings.encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        self.classes: List[str] = list(self.label_encoder.classes_)
        self.interpreter: Any = Interpreter(model_path=settings.model_path)
        self.interpreter.allocate_tensors()
        self._in_idx = self.interpreter.get_input_details()[0]["index"]
        self._out_idx = self.interpreter.get_output_details()[0]["index"]

    def predict(self, mel_features: np.ndarray) -> InferenceResult:
        """mel_features: shape (n_mels, n_frames, 1), as returned by
        signal_pipeline.extract_mel -- batch dimension added here."""
        batched = mel_features[np.newaxis, ...]
        self.interpreter.set_tensor(self._in_idx, batched)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self._out_idx)[0]

        # Murmur-first decision rule: this is a deliberate safety choice,
        # not a bug. The model was tuned via threshold sweep to maximize
        # murmur recall (100% on the validation set) at some cost to
        # overall accuracy (75%), because a missed murmur is far more
        # dangerous than a false positive in a screening context. Do not
        # "simplify" this back to a plain argmax without re-reading
        # docs/MODEL_CARD.md first.
        murmur_p = float(probs[self.settings.murmur_idx])
        if murmur_p >= self.settings.murmur_threshold:
            pred_idx = self.settings.murmur_idx
        else:
            pred_idx = int(np.argmax(probs))

        pred = self.classes[pred_idx]
        reasoning = format_reasoning(pred, murmur_p, self.settings.murmur_threshold)

        return InferenceResult(
            pred=pred,
            probs=probs.tolist(),
            classes=self.classes,
            murmur_p=murmur_p,
            reasoning=reasoning,
        )
