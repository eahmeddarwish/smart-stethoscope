"""The three screens of the touchscreen app: loading -> live recording ->
result. Each screen only builds UI and exposes update methods; all
business logic (recording, inference, PDF export) lives in app.py and the
non-Qt modules so it can be tested without a display."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .widgets import DemoFilePreview, MplCanvas, RecordingProgressWidget


class LoadingScreen(QWidget):
    def __init__(self, done_callback):
        super().__init__()
        self.setStyleSheet(f"background:{theme.BG};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(30)
        lay.setAlignment(Qt.AlignCenter)

        lay.addWidget(theme.lbl("SMART STETHOSCOPE", theme.CYAN, 28, bold=True, align=Qt.AlignCenter))
        lay.addWidget(theme.lbl("Initializing AI Engine...", theme.WHITE, 14, align=Qt.AlignCenter))
        lay.addWidget(theme.lbl("smart-stethoscope", theme.GRAY, 11, align=Qt.AlignCenter))

        QTimer.singleShot(1200, done_callback)


class LiveScreen(QWidget):
    """Recording screen: BPM + probability bars on the left, record button
    and demo/mic status in the center, mode/power controls on the right."""

    def __init__(self, murmur_threshold: float, noise_gate: float, duration: int, record_sec: int, sample_rate: int):
        super().__init__()
        self.setStyleSheet(f"background:{theme.BG};")
        self._setup_ui(murmur_threshold, noise_gate, duration, record_sec, sample_rate)

    def _setup_ui(self, murmur_threshold, noise_gate, duration, record_sec, sample_rate):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet(f"background:{theme.BG2};border-bottom:2px solid {theme.BORDER};")
        hlay = QHBoxLayout(header)
        hlay.setContentsMargins(14, 0, 14, 0)
        hlay.addWidget(theme.lbl("AI SMART STETHOSCOPE", theme.CYAN, 18, bold=True))
        self.mode_badge = theme.lbl("[ DEMO MODE ]", theme.ORANGE, 11, bold=True)
        hlay.addWidget(self.mode_badge)
        hlay.addStretch()
        self.mode_lbl = theme.lbl("Mode: DEMO", theme.WHITE, 12)
        hlay.addWidget(self.mode_lbl)
        self.pt_time_lbl = theme.lbl("--:--:--", theme.GRAY, 11)
        hlay.addWidget(self.pt_time_lbl)
        root.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(10, 10, 10, 10)
        body.setSpacing(10)

        # LEFT: BPM + probabilities
        left = QVBoxLayout()
        left.setSpacing(10)

        bpm_frame = QFrame()
        bpm_frame.setFixedWidth(170)
        bpm_frame.setStyleSheet(f"background:{theme.BG2};border:2px solid {theme.BORDER};border-radius:8px;")
        bpm_lay = QVBoxLayout(bpm_frame)
        bpm_lay.setAlignment(Qt.AlignCenter)
        self.bpm_lbl = theme.lbl("--", theme.CYAN, 36, bold=True, align=Qt.AlignCenter)
        bpm_lay.addWidget(self.bpm_lbl)
        bpm_lay.addWidget(theme.lbl("BPM", theme.GRAY, 12, align=Qt.AlignCenter))
        left.addWidget(bpm_frame)

        prob_frame = QFrame()
        prob_frame.setFixedWidth(170)
        prob_frame.setStyleSheet(f"background:{theme.BG2};border:2px solid {theme.BORDER};border-radius:8px;")
        prob_lay = QVBoxLayout(prob_frame)
        prob_lay.setSpacing(7)
        prob_lay.addWidget(theme.lbl("PROBABILITIES", theme.GRAY, 13, bold=True))

        self.prob_bars = {}
        self.prob_lbls = {}
        for cls in ["artifact", "extrahls", "murmur", "normal"]:
            row = QHBoxLayout()
            row.setSpacing(5)
            name_l = theme.lbl(cls[:4].upper(), theme.GRAY, 10, bold=True)
            name_l.setFixedWidth(35)
            row.addWidget(name_l)
            bar = QFrame()
            bar.setFixedHeight(12)
            bar.setStyleSheet(f"background:{theme.BG3};border-radius:4px;")
            bar_inner = QFrame(bar)
            bar_inner.setFixedHeight(12)
            bar_inner.setStyleSheet(f"background:{theme.CLASS_COLOR[cls]};border-radius:4px;")
            bar_inner.setFixedWidth(0)
            self.prob_bars[cls] = bar_inner
            row.addWidget(bar)
            pct_l = theme.lbl("0%", theme.WHITE, 10, bold=True)
            pct_l.setFixedWidth(32)
            self.prob_lbls[cls] = pct_l
            row.addWidget(pct_l)
            prob_lay.addLayout(row)

        left.addWidget(prob_frame)
        left.addStretch()
        body.addLayout(left)

        # CENTER: demo preview, recording progress, mel, result banner, record button
        center = QVBoxLayout()
        center.setSpacing(8)

        self.demo_preview = DemoFilePreview()
        center.addWidget(self.demo_preview)

        self.wave = RecordingProgressWidget(record_sec=record_sec)
        center.addWidget(self.wave)

        self.mel_canvas = MplCanvas(rows=1, cols=1, figsize=(5.5, 1.8))
        self.mel_canvas.setFixedHeight(120)
        self.mel_canvas.clear("Mel Spectrogram")
        center.addWidget(self.mel_canvas)

        self.result_frame = QFrame()
        self.result_frame.setFixedHeight(48)
        self.result_frame.setStyleSheet(f"background:{theme.BG3};border:2px solid {theme.BORDER};border-radius:8px;")
        res_lay = QHBoxLayout(self.result_frame)
        self.result_lbl = theme.lbl("Ready to record", theme.GRAY, 16, bold=True, align=Qt.AlignCenter)
        res_lay.addWidget(self.result_lbl, alignment=Qt.AlignCenter)
        center.addWidget(self.result_frame)

        self.rec_btn = QPushButton("[ STETHOSCOPE ]  RECORD  (%ds)" % record_sec)
        self.rec_btn.setFixedHeight(56)
        self.rec_btn.setStyleSheet(
            f"background:{theme.RED};color:{theme.WHITE};font-size:16px;font-weight:bold;"
            f"border-radius:10px;border:none;letter-spacing:1px;"
        )
        center.addWidget(self.rec_btn)
        body.addLayout(center, stretch=1)

        # RIGHT: power / mode toggle / pipeline settings
        right = QVBoxLayout()
        right.setSpacing(10)

        pw_frame = QFrame()
        pw_frame.setFixedWidth(165)
        pw_frame.setStyleSheet(f"background:{theme.BG2};border:2px solid {theme.BORDER};border-radius:8px;")
        pw_lay = QVBoxLayout(pw_frame)
        pw_lay.setSpacing(8)
        pw_lay.addWidget(theme.lbl("POWER", theme.GRAY, 10, bold=True))
        self.power_btn = QPushButton("[  ] SYSTEM ON")
        self.power_btn.setFixedHeight(42)
        self.power_btn.setStyleSheet(
            f"background:#0a2a18;color:{theme.GREEN};font-size:13px;font-weight:bold;"
            f"border:1px solid #1a5a30;border-radius:6px;"
        )
        pw_lay.addWidget(self.power_btn)
        right.addWidget(pw_frame)

        mode_frame = QFrame()
        mode_frame.setFixedWidth(165)
        mode_frame.setStyleSheet(f"background:{theme.BG2};border:2px solid {theme.BORDER};border-radius:8px;")
        mode_lay = QVBoxLayout(mode_frame)
        mode_lay.setSpacing(8)
        mode_lay.addWidget(theme.lbl("RECORDING MODE", theme.GRAY, 10, bold=True))
        self.live_btn = QPushButton("[*] Live Stethoscope")
        self.live_btn.setFixedHeight(38)
        self.demo_btn = QPushButton("[ ] Demo / Simulation")
        self.demo_btn.setFixedHeight(38)
        mode_lay.addWidget(self.live_btn)
        mode_lay.addWidget(self.demo_btn)
        right.addWidget(mode_frame)

        self.pdf_btn = QPushButton("[PDF] Export Report")

        cfg_frame = QFrame()
        cfg_frame.setFixedWidth(165)
        cfg_frame.setStyleSheet(f"background:{theme.BG2};border:2px solid {theme.BORDER};border-radius:8px;")
        cfg_lay = QVBoxLayout(cfg_frame)
        cfg_lay.setSpacing(4)
        cfg_lay.addWidget(theme.lbl("PIPELINE", theme.GRAY, 10, bold=True))
        for line in [
            f"Threshold : {murmur_threshold:.2f}",
            f"Noise gate: {int(noise_gate * 100)}%",
            f"Window    : best {duration}s/{record_sec}s",
            f"SR        : {sample_rate} Hz",
            "Window sel: Peak Regularity",
        ]:
            cfg_lay.addWidget(theme.lbl(line, theme.GRAY, 9))
        right.addWidget(cfg_frame)

        right.addStretch()
        body.addLayout(right)
        root.addLayout(body, stretch=1)

    def set_status(self, msg, color=theme.GRAY):
        self.result_lbl.setText(msg)
        self.result_frame.setStyleSheet(f"background:{theme.BG3};border:2px solid {theme.BORDER};border-radius:8px;")
        self.result_lbl.setStyleSheet(f"color:{color};font-size:14px;font-weight:bold;background:transparent;")

    def set_result(self, pred, conf_pct, color, title):
        self.result_lbl.setText(f"{title}  --  {conf_pct:.1f}%")
        self.result_frame.setStyleSheet(
            f"background:{theme.BANNER_BG.get(pred, theme.BG3)};"
            f"border:2px solid {theme.BANNER_BORDER.get(pred, theme.BORDER)};border-radius:8px;"
        )
        self.result_lbl.setStyleSheet(f"color:{color};font-size:14px;font-weight:bold;background:transparent;")

    def set_busy(self, msg, color=theme.CYAN):
        self.set_status(msg, color)
        self.rec_btn.setEnabled(False)

    def set_ready(self):
        self.rec_btn.setEnabled(True)

    def update_probs(self, probs, classes):
        bar_w = 100
        for i, cls in enumerate(classes):
            p = float(probs[i])
            w = int(p * bar_w)
            if cls in self.prob_bars:
                self.prob_bars[cls].setFixedWidth(w)
                self.prob_lbls[cls].setText(f"{p * 100:.0f}%")

    def update_bpm(self, bpm):
        self.bpm_lbl.setText(str(bpm))

    def update_mode_display(self, test_mode: bool):
        if test_mode:
            self.mode_badge.setText("[ DEMO MODE ]")
            self.mode_badge.setStyleSheet(f"color:{theme.ORANGE};font-size:11px;background:transparent;font-weight:bold;")
            self.mode_lbl.setText("Mode: DEMO")
            self.rec_btn.setText("[ DEMO ]  RECORD (Simulation)")
            self.live_btn.setText("[ ] Live Stethoscope")
            self.demo_btn.setText("[*] Demo / Simulation")
            self.demo_preview.show()
        else:
            self.mode_badge.setText("[ LIVE MODE ]")
            self.mode_badge.setStyleSheet(f"color:{theme.GREEN};font-size:11px;background:transparent;font-weight:bold;")
            self.mode_lbl.setText("Mode: LIVE")
            self.rec_btn.setText("[ STETHOSCOPE ]  RECORD")
            self.live_btn.setText("[*] Live Stethoscope")
            self.demo_btn.setText("[ ] Demo / Simulation")
            self.demo_preview.hide_preview()


class ResultScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background:{theme.BG};")
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(52)
        header.setStyleSheet(f"background:{theme.BG2};border-bottom:2px solid {theme.BORDER};")
        hlay = QHBoxLayout(header)
        hlay.addWidget(theme.lbl("[R] Diagnosis Report", theme.CYAN, 18, bold=True))
        self.header_info = theme.lbl("", theme.GRAY, 11)
        hlay.addWidget(self.header_info)
        hlay.addStretch()
        root.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(12, 10, 12, 10)
        center = QVBoxLayout()
        center.setSpacing(8)

        self.banner = QFrame()
        self.banner.setFixedHeight(52)
        self.banner.setStyleSheet(f"background:{theme.BG3};border:2px solid {theme.BORDER};border-radius:10px;")
        ban_lay = QHBoxLayout(self.banner)
        self.banner_lbl = theme.lbl("", theme.WHITE, 20, bold=True, align=Qt.AlignCenter)
        ban_lay.addWidget(self.banner_lbl, alignment=Qt.AlignCenter)
        center.addWidget(self.banner)

        plots_row = QHBoxLayout()
        plots_row.setSpacing(10)
        self.wave_canvas = MplCanvas(rows=1, cols=1, figsize=(4.5, 2))
        self.wave_canvas.setFixedHeight(145)
        plots_row.addWidget(self.wave_canvas)
        self.mel_canvas = MplCanvas(rows=1, cols=1, figsize=(4.5, 2))
        self.mel_canvas.setFixedHeight(145)
        plots_row.addWidget(self.mel_canvas)
        center.addLayout(plots_row)

        reasoning_frame = QFrame()
        reasoning_frame.setStyleSheet(f"background:{theme.BG2};border:2px solid {theme.BORDER};border-radius:8px;")
        reason_lay = QVBoxLayout(reasoning_frame)
        reason_lay.addWidget(theme.lbl("CLINICAL REASONING:", theme.CYAN, 12, bold=True))
        self.reasoning_lbl = QLabel()
        self.reasoning_lbl.setWordWrap(True)
        self.reasoning_lbl.setStyleSheet(f"color:{theme.WHITE};font-size:15px;background:transparent;")
        reason_lay.addWidget(self.reasoning_lbl)
        center.addWidget(reasoning_frame)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.back_btn = QPushButton("< Back")
        self.back_btn.setFixedHeight(54)
        self.pdf_btn = QPushButton("[PDF]  Export Hospital Report")
        self.pdf_btn.setFixedHeight(54)
        self.play_btn = QPushButton("[>] Play Audio")
        self.play_btn.setFixedHeight(54)
        self.repeat_btn = QPushButton("[*] Record Again")
        self.repeat_btn.setFixedHeight(54)
        btn_row.addWidget(self.back_btn)
        btn_row.addWidget(self.pdf_btn, stretch=1)
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.repeat_btn)
        center.addLayout(btn_row)

        body.addLayout(center, stretch=1)
        root.addLayout(body, stretch=1)

    def load_data(self, pred, title, color, murmur_p, patient_id, timestamp, mode, audio, reasoning,
                  sample_rate, n_mels, n_fft, hop_length, duration):
        self.banner_lbl.setText(f"{title}  --  Murmur: {murmur_p * 100:.1f}%")
        self.banner.setStyleSheet(
            f"background:{theme.BANNER_BG.get(pred, theme.BG3)};"
            f"border:2px solid {theme.BANNER_BORDER.get(pred, theme.BORDER)};border-radius:10px;"
        )
        self.banner_lbl.setStyleSheet(f"color:{color};font-size:18px;font-weight:bold;background:transparent;")
        self.header_info.setText(f"{patient_id}  |  {timestamp}  |  {mode}")
        self.wave_canvas.plot_wave(audio, color=color, title="Enhanced Waveform", duration=duration)
        self.mel_canvas.plot_mel(
            audio, sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, duration=duration
        )
        self.reasoning_lbl.setText(reasoning[:400] + ("..." if len(reasoning) > 400 else ""))
