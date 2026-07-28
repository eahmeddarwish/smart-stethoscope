"""Reusable Qt widgets: matplotlib canvas, recording progress bar, demo
file preview thumbnail."""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QSizePolicy, QVBoxLayout, QWidget

from . import theme


class MplCanvas(FigureCanvas):
    def __init__(self, rows: int = 1, cols: int = 1, figsize=(4, 2)):
        self.fig = Figure(figsize=figsize, facecolor=theme.BG2)
        self.axes = []
        for i in range(rows * cols):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self._style_ax(ax)
            self.axes.append(ax)
        self.fig.tight_layout(pad=0.5)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(theme.BG)
        ax.tick_params(colors=theme.GRAY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(theme.BORDER)
            spine.set_linewidth(1.5)

    def clear_all(self):
        for ax in self.axes:
            ax.clear()
            self._style_ax(ax)

    def plot_wave(self, audio, ax_idx=0, color=theme.CYAN, title="", duration=3):
        ax = self.axes[ax_idx]
        ax.clear()
        self._style_ax(ax)
        t = np.linspace(0, duration, len(audio))
        ax.plot(t, audio, color=color, linewidth=1.2)
        ax.set_title(title, color=theme.WHITE, fontsize=10, pad=4, fontweight="bold")
        ax.set_xlabel("Time (s)", color=theme.GRAY, fontsize=9)
        ax.set_ylabel("Amplitude", color=theme.GRAY, fontsize=9)
        ax.grid(True, alpha=0.15, color=theme.GRAY)
        ax.set_xlim([0, duration])
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def plot_mel(self, audio, ax_idx=0, sample_rate=2000, n_mels=64, n_fft=512, hop_length=128, duration=3):
        import librosa

        ax = self.axes[ax_idx]
        ax.clear()
        self._style_ax(ax)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        ax.imshow(mel_db, aspect="auto", origin="lower", cmap="inferno", extent=[0, duration, 0, n_mels])
        ax.set_title("Mel Spectrogram", color=theme.WHITE, fontsize=10, pad=4, fontweight="bold")
        ax.set_xlabel("Time (s)", color=theme.GRAY, fontsize=9)
        ax.set_ylabel("Mel Band", color=theme.GRAY, fontsize=9)
        self.fig.tight_layout(pad=0.5)
        self.draw()

    def clear(self, msg=""):
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(theme.BG)
            if msg:
                ax.text(0.5, 0.5, msg, transform=ax.transAxes, color=theme.GRAY, ha="center", va="center", fontsize=11)
            for spine in ax.spines.values():
                spine.set_edgecolor(theme.BORDER)
                spine.set_linewidth(1.5)
        self.draw()


class RecordingProgressWidget(QWidget):
    """Animated progress bar shown while the mic (or demo load) is running."""

    def __init__(self, record_sec: int = 8):
        super().__init__()
        self.total = record_sec
        self.setFixedHeight(140)
        self.setStyleSheet(f"background:{theme.BG2};border:1px solid {theme.BORDER};border-radius:8px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        self.title_lbl = theme.lbl("Ready to record", theme.CYAN, 14, bold=True, align=Qt.AlignCenter)
        lay.addWidget(self.title_lbl)

        bar_frame = QFrame()
        bar_frame.setFixedHeight(28)
        bar_frame.setStyleSheet(f"background:{theme.BG3};border:1px solid {theme.BORDER};border-radius:6px;")
        self.bar_inner = QFrame(bar_frame)
        self.bar_inner.setFixedHeight(28)
        self.bar_inner.setFixedWidth(0)
        self.bar_inner.setStyleSheet(f"background:{theme.CYAN};border-radius:6px;")
        lay.addWidget(bar_frame)

        self.time_lbl = theme.lbl(f"0.0 / {self.total:.1f} s", theme.WHITE, 13, bold=True, align=Qt.AlignCenter)
        lay.addWidget(self.time_lbl)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.elapsed = 0.0

    def start_anim(self):
        self.elapsed = 0.0
        self.bar_inner.setFixedWidth(0)
        self.title_lbl.setText("Recording in progress...")
        self.title_lbl.setStyleSheet(f"color:{theme.RED};font-size:14px;font-weight:bold;background:transparent;")
        self.timer.start(100)

    def stop_anim(self):
        self.timer.stop()
        self.title_lbl.setText("Processing audio...")
        self.title_lbl.setStyleSheet(f"color:{theme.ORANGE};font-size:14px;font-weight:bold;background:transparent;")

    def _tick(self):
        self.elapsed += 0.1
        if self.elapsed > self.total:
            self.stop_anim()
            return
        frac = self.elapsed / self.total
        w = int(frac * (self.width() - 36))
        self.bar_inner.setFixedWidth(w)
        self.time_lbl.setText(f"{self.elapsed:.1f} / {self.total:.1f} s")


class DemoFilePreview(QFrame):
    """Shows which archived WAV is being used in demo/simulation mode --
    keeps demo runs clearly distinguishable from a live capture."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(90)
        self.setStyleSheet(f"background:{theme.BG2};border:1px solid {theme.BORDER};border-radius:8px;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(6)

        self.file_lbl = theme.lbl("No file loaded", theme.ORANGE, 12, bold=True, align=Qt.AlignCenter)
        lay.addWidget(self.file_lbl)

        self.preview_canvas = MplCanvas(rows=1, cols=1, figsize=(6, 0.8))
        self.preview_canvas.setFixedHeight(50)
        self.preview_canvas.clear("Waveform Preview")
        lay.addWidget(self.preview_canvas)

        self.hide()

    def show_file(self, filename, audio_preview=None):
        self.file_lbl.setText(f"DEMO FILE: {filename}")
        if audio_preview is not None and len(audio_preview) > 0:
            self.preview_canvas.plot_wave(audio_preview, color=theme.ORANGE, title="")
        self.show()

    def hide_preview(self):
        self.file_lbl.setText("No file loaded")
        self.preview_canvas.clear()
        self.hide()
