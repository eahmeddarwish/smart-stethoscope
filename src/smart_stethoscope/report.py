"""
PDF diagnosis report generation (ReportLab), decoupled from Qt so it can
be unit-tested and re-used from a CLI or batch job.
"""
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import Settings
from .inference import InferenceResult

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdf_canvas

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

CLASS_COLORS = {
    "artifact": (0.3, 0.45, 0.65),
    "extrahls": (0.9, 0.75, 0.1),
    "murmur": (0.9, 0.2, 0.2),
    "normal": (0.1, 0.75, 0.3),
}


def generate_pdf_report(
    settings: Settings,
    result: InferenceResult,
    audio: np.ndarray,
    raw_2k: np.ndarray,
    filtered: np.ndarray,
    bpm: int,
    source: str,
    patient_id: str = "DEMO-PATIENT-001",
    mode: str = "DEMO",
) -> Optional[str]:
    """Writes a two-page PDF report into settings.reports_dir and returns
    its path, or None (and logs why) if ReportLab isn't installed."""
    if not REPORTLAB_AVAILABLE:
        print("reportlab not installed -- run `pip install reportlab` to enable PDF export.")
        return None

    ts_safe = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(settings.reports_dir, f"report_{ts_safe}.pdf")

    W, H = A4
    margin = 2 * cm
    c = pdf_canvas.Canvas(out_path, pagesize=A4)

    def draw_dark_bg():
        c.setFillColorRGB(0.04, 0.06, 0.10)
        c.rect(0, 0, W, H, fill=1, stroke=0)

    def section_title(txt, y):
        c.setFont("Helvetica-Bold", 18)
        c.setFillColorRGB(0, 0.9, 1.0)
        c.drawString(margin, y, txt)
        c.setLineWidth(1.5)
        c.setStrokeColorRGB(0.12, 0.19, 0.31)
        c.line(margin, y - 6, W - margin, y - 6)
        return y - 24

    def field_row(label, val, y, col=(0.8, 0.9, 1.0)):
        c.setFont("Helvetica-Bold", 14)
        c.setFillColorRGB(0.4, 0.55, 0.7)
        c.drawString(margin, y, label + ":")
        c.setFont("Helvetica", 14)
        c.setFillColorRGB(*col)
        c.drawString(margin + 130, y, str(val))
        return y - 20

    def wrap_text(txt, y, color=(0.7, 0.8, 0.9), line_h=18):
        c.setFont("Helvetica", 14)
        c.setFillColorRGB(*color)
        max_w = W - 2 * margin - 10
        words = txt.split()
        line = ""
        for w_txt in words:
            test = line + " " + w_txt if line else w_txt
            if c.stringWidth(test, "Helvetica", 14) < max_w:
                line = test
            else:
                c.drawString(margin, y, line)
                y -= line_h
                line = w_txt
                if y < margin + 40:
                    c.showPage()
                    draw_dark_bg()
                    y = H - margin
        if line:
            c.drawString(margin, y, line)
            y -= line_h
        return y

    def embed_mpl_fig(fig, x, y, w, h):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor="#0a0e1a", bbox_inches="tight")
        buf.seek(0)
        c.drawImage(ImageReader(buf), x, y - h, w, h)

    pred = result.pred
    pred_color = {
        "murmur": (1, 0.25, 0.25),
        "normal": (0.2, 0.8, 0.35),
        "extrahls": (1, 0.8, 0.1),
    }.get(pred, (0.5, 0.65, 0.8))

    # ---- PAGE 1: patient info, probabilities, findings ----
    draw_dark_bg()
    c.setFillColorRGB(0.08, 0.13, 0.42)
    c.rect(0, H - 80, W, 80, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, H - 35, "AI Smart Stethoscope -- Diagnosis Report")
    c.setFont("Helvetica", 12)
    c.setFillColorRGB(0.6, 0.7, 0.9)
    c.drawString(margin, H - 55, "Kuwait College of Science & Technology -- Group 07")
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.4, 0.5, 0.7)
    c.drawString(margin, H - 72, "AI-generated report. Does not replace professional medical diagnosis.")

    y = H - 100
    y = section_title("PATIENT & SESSION INFORMATION", y)

    source_display = os.path.basename(source) if source else "LIVE MIC"
    from .clinical_info import CLINICAL_INFO

    title = CLINICAL_INFO.get(pred, CLINICAL_INFO["artifact"])["title"]

    for label, val, col in [
        ("Patient ID", patient_id, (0.8, 0.9, 1.0)),
        ("Date & Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (0.8, 0.9, 1.0)),
        ("Recording Mode", mode, (0.8, 0.9, 1.0)),
        ("Source File", source_display, (0.8, 0.9, 1.0)),
        ("Diagnosis", title, pred_color),
        ("Murmur Prob", "%.1f%%" % (result.murmur_p * 100), pred_color),
        ("Murmur Threshold", "%.2f (conservative for safety)" % settings.murmur_threshold, (0.8, 0.9, 1.0)),
        ("Est. Heart Rate", "%d BPM" % bpm, (0, 0.9, 1.0)),
    ]:
        y = field_row(label, val, y, col)

    y -= 12
    y = section_title("CLASS PROBABILITIES", y)
    bar_max = W - 2 * margin - 180
    for i, cls in enumerate(result.classes):
        p = float(result.probs[i])
        c.setFont("Helvetica", 14)
        c.setFillColorRGB(0.5, 0.65, 0.8)
        c.drawString(margin, y, "%-10s" % cls.upper())
        c.setFillColorRGB(0.1, 0.15, 0.25)
        c.rect(margin + 100, y - 2, bar_max, 16, fill=1, stroke=0)
        col = CLASS_COLORS.get(cls, (0.5, 0.7, 0.9))
        c.setFillColorRGB(*col)
        if p > 0:
            c.rect(margin + 100, y - 2, p * bar_max, 16, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.drawString(margin + 100 + bar_max + 8, y, "%.1f%%" % (p * 100))
        y -= 24

    y -= 12
    y = section_title("CLINICAL FINDINGS", y)
    for reason in CLINICAL_INFO.get(pred, CLINICAL_INFO["artifact"])["reasons"]:
        c.setFont("Helvetica", 13)
        c.setFillColorRGB(0.65, 0.78, 0.88)
        c.drawString(margin + 12, y, "- " + reason)
        y -= 20

    y -= 12
    y = section_title("DETAILED CLINICAL REASONING", y)
    y = wrap_text(result.reasoning, y)

    y -= 12
    y = section_title("MEDICAL RECOMMENDATION", y)
    y = wrap_text(CLINICAL_INFO.get(pred, CLINICAL_INFO["artifact"])["recommendation"], y, color=(0.2, 0.85, 1.0))

    # ---- PAGE 2: signal plots ----
    c.showPage()
    draw_dark_bg()
    c.setFillColorRGB(0.08, 0.13, 0.42)
    c.rect(0, H - 65, W, 65, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - 30, "Signal Analysis -- Waveforms & Spectrograms")

    plot_y = H - 85
    plot_h = 3.8 * cm
    plot_w = (W - 2 * margin - cm) / 2

    fig1, ax1 = plt.subplots(figsize=(5.5, 2.5), facecolor="#0a0e1a")
    ax1.set_facecolor("#0a0e1a")
    t_full = np.linspace(0, settings.record_sec, len(raw_2k))
    ax1.plot(t_full, raw_2k, color="#3a6080", linewidth=0.8, alpha=0.7, label="Raw 2kHz")
    ax1.plot(t_full, filtered, color="#00aacc", linewidth=0.9, alpha=0.8, label="Filtered")
    ax1.set_title("Full Recording (Filtered)", color="#7aaac8", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="#4a7090", labelsize=8)
    ax1.legend(fontsize=8, facecolor="#0d1829", labelcolor="#7aaac8")
    fig1.tight_layout(pad=0.4)
    embed_mpl_fig(fig1, margin, plot_y, plot_w, plot_h)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(5.5, 2.5), facecolor="#0a0e1a")
    ax2.set_facecolor("#0a0e1a")
    t_clip = np.linspace(0, settings.duration, len(audio))
    ax2.plot(t_clip, audio, color="#00e5ff", linewidth=1.1)
    ax2.set_title("Enhanced Heart Sound (Best Window)", color="#7aaac8", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="#4a7090", labelsize=8)
    fig2.tight_layout(pad=0.4)
    embed_mpl_fig(fig2, margin + plot_w + cm, plot_y, plot_w, plot_h)
    plt.close(fig2)

    plot_y -= plot_h + 0.6 * cm

    import librosa

    fig3, ax3 = plt.subplots(figsize=(5.5, 2.5), facecolor="#0a0e1a")
    ax3.set_facecolor("#0a0e1a")
    mel = librosa.feature.melspectrogram(
        y=audio, sr=settings.sample_rate, n_mels=settings.n_mels,
        n_fft=settings.n_fft, hop_length=settings.hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    ax3.imshow(mel_db, aspect="auto", origin="lower", cmap="inferno",
               extent=[0, settings.duration, 0, settings.n_mels])
    ax3.set_title("Mel Spectrogram (Model Input)", color="#7aaac8", fontsize=11, fontweight="bold")
    ax3.tick_params(colors="#4a7090", labelsize=8)
    fig3.tight_layout(pad=0.4)
    embed_mpl_fig(fig3, margin, plot_y, plot_w, plot_h)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(5.5, 2.5), facecolor="#0a0e1a")
    ax4.set_facecolor("#0a0e1a")
    N = len(audio)
    freqs = np.fft.rfftfreq(N, d=1.0 / settings.sample_rate)
    fft_m = 20 * np.log10(np.abs(np.fft.rfft(audio)) / N + 1e-9)
    ax4.plot(freqs, fft_m, color="#00cc88", linewidth=1.0)
    ax4.axvspan(20, 150, alpha=0.15, color="#00cc88", label="S1/S2 (20-150Hz)")
    ax4.axvspan(150, 400, alpha=0.10, color="#ffcc00", label="Murmur (150-400Hz)")
    ax4.set_title("Frequency Spectrum (dB)", color="#7aaac8", fontsize=11, fontweight="bold")
    ax4.set_xlim([0, 500])
    ax4.tick_params(colors="#4a7090", labelsize=8)
    ax4.legend(fontsize=8, facecolor="#0d1829", labelcolor="#7aaac8")
    fig4.tight_layout(pad=0.4)
    embed_mpl_fig(fig4, margin + plot_w + cm, plot_y, plot_w, plot_h)
    plt.close(fig4)

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.3, 0.4, 0.55)
    c.drawString(margin, margin / 2,
                 "AI Smart Stethoscope -- KCST Group 07 | AI-generated, does not replace professional diagnosis.")

    c.save()
    return out_path
