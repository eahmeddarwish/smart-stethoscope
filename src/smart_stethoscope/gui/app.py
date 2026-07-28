"""
MainWindow: wires the three screens together with the non-Qt business
logic (audio_capture, signal_pipeline, inference, report). Threading is
kept minimal and explicit -- recording and inference each run on a plain
daemon Thread, callbacks are marshalled back to the Qt thread with Qt
signals only.
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime

import librosa
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from ..audio_capture import (
    SOUNDDEVICE_AVAILABLE,
    load_demo_wav,
    pick_random_demo_wav,
    record_from_microphone,
)
from ..clinical_info import CLINICAL_INFO
from ..config import SETTINGS, Settings
from ..inference import HeartSoundClassifier, InferenceResult
from ..report import REPORTLAB_AVAILABLE, generate_pdf_report
from ..signal_pipeline import estimate_bpm, extract_mel
from . import theme
from .screens import LiveScreen, LoadingScreen, ResultScreen

try:
    import sounddevice as sd
except ImportError:
    sd = None


class _AudioReady(QObject):
    ready = Signal(object)


def _play_audio_async(audio, sample_rate, on_done):
    def _run():
        try:
            sd.play(audio, sample_rate)
            sd.wait()
            on_done(True)
        except Exception:
            on_done(False)

    threading.Thread(target=_run, daemon=True).start()


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings = SETTINGS):
        super().__init__()
        self.settings = settings
        self.setWindowTitle("AI Smart Stethoscope")
        self.showFullScreen()

        self.test_mode = True
        self.power_on = True
        self.busy = False
        self.current_demo_file = None
        self.last_result_data = None

        self.classifier = None
        try:
            self.classifier = HeartSoundClassifier(settings)
        except Exception as exc:  # model not present yet, or wrong path
            print(f"[startup] Could not load model: {exc}")
            print(f"  Expected at: {settings.model_path}")
            print("  Set STETHOSCOPE_BASE_DIR or copy the model into that folder.")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.s_load = LoadingScreen(self._on_loaded)
        self.s_live = LiveScreen(
            murmur_threshold=settings.murmur_threshold,
            noise_gate=settings.noise_gate,
            duration=settings.duration,
            record_sec=settings.record_sec,
            sample_rate=settings.sample_rate,
        )
        self.s_result = ResultScreen()

        self.stack.addWidget(self.s_load)
        self.stack.addWidget(self.s_live)
        self.stack.addWidget(self.s_result)
        self.stack.setCurrentIndex(0)

        self.s_live.rec_btn.clicked.connect(self.on_record)
        self.s_live.power_btn.clicked.connect(self._toggle_power)
        self.s_live.live_btn.clicked.connect(self._set_live_mode)
        self.s_live.demo_btn.clicked.connect(self._set_demo_mode)
        self.s_live.pdf_btn.clicked.connect(self._save_pdf)

        self.s_result.back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.s_result.play_btn.clicked.connect(self._play_audio)
        self.s_result.pdf_btn.clicked.connect(self._save_pdf)
        self.s_result.repeat_btn.clicked.connect(self._repeat)

        self._audio_ready = _AudioReady()
        self._audio_ready.ready.connect(self._on_audio_captured)

    # ---- lifecycle ----------------------------------------------------
    def _on_loaded(self):
        self.s_live.update_mode_display(self.test_mode)
        self.stack.setCurrentIndex(1)

    def _toggle_power(self):
        self.power_on = not self.power_on
        if self.power_on:
            self.s_live.power_btn.setText("[  ] SYSTEM ON")
            self.s_live.rec_btn.setEnabled(True)
        else:
            self.s_live.power_btn.setText("[X] SYSTEM OFF")
            self.s_live.rec_btn.setEnabled(False)
            self.s_live.wave.stop_anim()
            self.s_live.set_status("System powered off", theme.RED)

    def _set_live_mode(self):
        self.test_mode = False
        self.s_live.update_mode_display(self.test_mode)

    def _set_demo_mode(self):
        self.test_mode = True
        self.s_live.update_mode_display(self.test_mode)

    # ---- recording ------------------------------------------------------
    def on_record(self):
        if self.busy or self.classifier is None or not self.power_on:
            return
        self.busy = True
        self.s_live.pt_time_lbl.setText(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        self.s_live.wave.start_anim()

        if self.test_mode:
            wav_path = pick_random_demo_wav(self.settings)
            if not wav_path:
                self.s_live.set_status("No WAV files found under dataset_dir", theme.RED)
                self.busy = False
                self.s_live.wave.stop_anim()
                return
            self.current_demo_file = wav_path
            filename = wav_path.split("/")[-1].split("\\")[-1]
            self.s_live.set_busy(f"Loading: {filename}", theme.CYAN)
            try:
                preview_audio, _ = librosa.load(wav_path, sr=self.settings.sample_rate, duration=0.5)
                self.s_live.demo_preview.show_file(filename, preview_audio)
            except Exception:
                self.s_live.demo_preview.show_file(filename, None)
            threading.Thread(target=self._load_wav, args=(wav_path,), daemon=True).start()
        else:
            if not SOUNDDEVICE_AVAILABLE:
                self.s_live.set_status("sounddevice not available", theme.RED)
                self.busy = False
                self.s_live.wave.stop_anim()
                return
            self.current_demo_file = None
            self.s_live.demo_preview.hide_preview()
            self.s_live.set_busy(f"Recording {self.settings.record_sec}s...", theme.RED)
            threading.Thread(target=self._record_mic, daemon=True).start()

    def _load_wav(self, path):
        try:
            captured = load_demo_wav(path, self.settings)
            self._audio_ready.ready.emit(captured)
        except Exception as exc:
            self._on_capture_error(str(exc))

    def _record_mic(self):
        try:
            captured = record_from_microphone(self.settings)
            self._audio_ready.ready.emit(captured)
        except Exception as exc:
            self._on_capture_error(str(exc))

    def _on_capture_error(self, msg: str):
        self.s_live.set_status("Error: " + msg, theme.RED)
        self.busy = False
        self.s_live.wave.stop_anim()
        self.s_live.set_ready()

    def _on_audio_captured(self, captured):
        self.s_live.wave.stop_anim()
        self.s_live.set_busy("Analyzing...", theme.CYAN)
        self.s_live.mel_canvas.plot_mel(
            captured.audio, sample_rate=self.settings.sample_rate, n_mels=self.settings.n_mels,
            n_fft=self.settings.n_fft, hop_length=self.settings.hop_length, duration=self.settings.duration,
        )
        threading.Thread(target=self._run_inference, args=(captured,), daemon=True).start()

    def _run_inference(self, captured):
        try:
            mel = extract_mel(captured.audio, self.settings)
            result: InferenceResult = self.classifier.predict(mel)
            bpm = estimate_bpm(captured.audio, self.settings.sample_rate, self.settings.duration)
            self.last_result_data = {
                "result": result,
                "captured": captured,
                "bpm": bpm,
                "patient_id": "DEMO-PATIENT-001",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "DEMO" if self.test_mode else "LIVE",
            }
            QTimer.singleShot(0, self._on_result_ready)
        except Exception as exc:
            QTimer.singleShot(0, lambda exc=exc: self._on_capture_error(str(exc)))

    def _on_result_ready(self):
        self.busy = False
        self.s_live.set_ready()
        d = self.last_result_data
        result: InferenceResult = d["result"]
        info = CLINICAL_INFO.get(result.pred, CLINICAL_INFO["artifact"])

        self.s_live.update_probs(result.probs, result.classes)
        self.s_live.update_bpm(d["bpm"])
        self.s_live.set_result(
            result.pred, result.murmur_p * 100, theme.CLASS_COLOR.get(result.pred, theme.WHITE), info["title"]
        )

        self.s_result.load_data(
            pred=result.pred, title=info["title"], color=theme.CLASS_COLOR.get(result.pred, theme.WHITE),
            murmur_p=result.murmur_p, patient_id=d["patient_id"], timestamp=d["timestamp"], mode=d["mode"],
            audio=d["captured"].audio, reasoning=result.reasoning,
            sample_rate=self.settings.sample_rate, n_mels=self.settings.n_mels,
            n_fft=self.settings.n_fft, hop_length=self.settings.hop_length, duration=self.settings.duration,
        )
        self.stack.setCurrentIndex(2)

    # ---- result screen actions -----------------------------------------
    def _repeat(self):
        self.stack.setCurrentIndex(1)
        QTimer.singleShot(150, self.on_record)

    def _play_audio(self):
        if not self.last_result_data or sd is None:
            return
        audio = self.last_result_data["captured"].audio
        btn = self.s_result.play_btn
        btn.setText("[||] Playing...")
        btn.setEnabled(False)

        def on_done(_ok):
            def _reset():
                btn.setText("[>] Play Audio")
                btn.setEnabled(True)

            QTimer.singleShot(0, _reset)

        _play_audio_async(audio, self.settings.sample_rate, on_done)

    def _save_pdf(self):
        if not self.last_result_data:
            self.s_live.set_status("No data yet -- record first", theme.ORANGE)
            return
        if not REPORTLAB_AVAILABLE:
            self.s_live.set_status("reportlab not installed", theme.RED)
            return
        self.s_live.set_status("Generating PDF report...", theme.CYAN)

        d = self.last_result_data
        c = d["captured"]

        def _generate():
            path = generate_pdf_report(
                self.settings, d["result"], c.audio, c.raw_2k, c.filtered, d["bpm"],
                source=c.source, patient_id=d["patient_id"], mode=d["mode"],
            )
            msg = f"PDF saved: {path}" if path else "PDF generation failed"
            color = theme.GREEN if path else theme.RED
            QTimer.singleShot(0, lambda: self.s_live.set_status(msg, color))

        threading.Thread(target=_generate, daemon=True).start()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(theme.BG))
    pal.setColor(QPalette.WindowText, QColor(theme.WHITE))
    pal.setColor(QPalette.Base, QColor(theme.BG2))
    pal.setColor(QPalette.Text, QColor(theme.WHITE))
    pal.setColor(QPalette.Button, QColor(theme.BG2))
    pal.setColor(QPalette.ButtonText, QColor(theme.WHITE))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
