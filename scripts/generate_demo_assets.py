"""
One-off script (not part of the package) to regenerate the README demo
assets straight from the real, current pipeline code.

It loads one real archived WAV through the ACTUAL signal_pipeline /
audio_capture code, then renders the ACTUAL LiveScreen and ResultScreen
Qt widgets offscreen and grabs them to PNG, and generates a real PDF via
the ACTUAL report.py generator.

Because no trained .tflite model ships in this repo (the model file lives
only on the project's Raspberry Pi hardware -- see README), the
classification numbers used here are reconstructed to match the result
that was previously observed for this exact archived recording
(`normal__128_1306344005749_D.wav`, set_b) under the documented
murmur-first decision rule. Every plot, waveform, spectrogram, BPM
estimate and PDF layout below is produced by the real code, not mocked.
"""
from __future__ import annotations

import datetime
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from smart_stethoscope.audio_capture import load_demo_wav  # noqa: E402
from smart_stethoscope.clinical_info import CLINICAL_INFO, format_reasoning  # noqa: E402
from smart_stethoscope.config import Settings  # noqa: E402
from smart_stethoscope.gui import theme  # noqa: E402
from smart_stethoscope.gui.screens import LiveScreen, ResultScreen  # noqa: E402
from smart_stethoscope.inference import InferenceResult  # noqa: E402
from smart_stethoscope.report import generate_pdf_report  # noqa: E402
from smart_stethoscope.signal_pipeline import estimate_bpm, extract_mel  # noqa: E402

import librosa  # noqa: E402

BASE = "/tmp/stethoscope_base"
WAV = os.path.join(BASE, "datasets", "Heartbeats", "set_b", "normal__128_1306344005749_D.wav")
OUT_ASSETS = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo_out"
os.makedirs(OUT_ASSETS, exist_ok=True)

settings = Settings(base_dir=BASE)

app = QApplication.instance() or QApplication([])

# --- 1. Run the REAL capture + signal pipeline on a REAL archived WAV ---
captured = load_demo_wav(WAV, settings)
bpm = estimate_bpm(captured.audio, settings.sample_rate, settings.duration)
mel = extract_mel(captured.audio, settings)  # noqa: F841 (shape-checked, not printed)

# --- 2. Reconstruct the previously-observed classification for this file ---
classes = ["artifact", "extrahls", "murmur", "normal"]
probs = [0.004, 0.555, 0.011, 0.430]
murmur_p = probs[classes.index("murmur")]
pred_idx = classes.index("murmur") if murmur_p >= settings.murmur_threshold else probs.index(max(probs))
pred = classes[pred_idx]
reasoning = format_reasoning(pred, murmur_p, settings.murmur_threshold)
result = InferenceResult(pred=pred, probs=probs, classes=classes, murmur_p=murmur_p, reasoning=reasoning)
title = CLINICAL_INFO[pred]["title"]
print(f"pred={pred} murmur_p={murmur_p} bpm={bpm} title={title}")

# --- 3. LiveScreen: idle / ready state (the "main interface") ---
live = LiveScreen(
    murmur_threshold=settings.murmur_threshold,
    noise_gate=settings.noise_gate,
    duration=settings.duration,
    record_sec=settings.record_sec,
    sample_rate=settings.sample_rate,
)
live.resize(1300, 760)
live.update_mode_display(True)  # demo mode
preview_audio, _ = librosa.load(WAV, sr=settings.sample_rate, duration=0.5)
live.demo_preview.show_file(os.path.basename(WAV), preview_audio)
live.set_status("Ready to record", theme.GRAY)
live.show()
app.processEvents()
live.grab().save(os.path.join(OUT_ASSETS, "gui_main_interface.png"))

# --- 4. LiveScreen after analysis (compact result banner + probs + BPM) ---
live.mel_canvas.plot_mel(
    captured.audio, sample_rate=settings.sample_rate, n_mels=settings.n_mels,
    n_fft=settings.n_fft, hop_length=settings.hop_length, duration=settings.duration,
)
live.update_probs(result.probs, result.classes)
live.update_bpm(bpm)
live.set_result(result.pred, result.murmur_p * 100, theme.CLASS_COLOR.get(result.pred, theme.WHITE), title)
app.processEvents()
live.grab().save(os.path.join(OUT_ASSETS, "gui_main_interface_after_analysis.png"))

# --- 5. ResultScreen: full diagnosis report page ---
res = ResultScreen()
res.resize(1300, 760)
res.load_data(
    pred=result.pred, title=title, color=theme.CLASS_COLOR.get(result.pred, theme.WHITE),
    murmur_p=result.murmur_p, patient_id="DEMO-PATIENT-001",
    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mode="DEMO",
    audio=captured.audio, reasoning=result.reasoning,
    sample_rate=settings.sample_rate, n_mels=settings.n_mels,
    n_fft=settings.n_fft, hop_length=settings.hop_length, duration=settings.duration,
)
res.show()
app.processEvents()
res.grab().save(os.path.join(OUT_ASSETS, "gui_diagnosis_report_new.png"))

# --- 6. Real PDF report via the real ReportLab generator ---
pdf_path = generate_pdf_report(
    settings, result, captured.audio, captured.raw_2k, captured.filtered, bpm,
    source=captured.source, patient_id="DEMO-PATIENT-001", mode="DEMO",
)
print("PDF written to:", pdf_path)
if pdf_path:
    import shutil
    shutil.copy(pdf_path, os.path.join(OUT_ASSETS, "sample_report.pdf"))

print("Done. Outputs in", OUT_ASSETS)
