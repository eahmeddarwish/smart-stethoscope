"""Shared color palette + a small QLabel factory used across all screens."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel

BG = "#0a0e1a"
BG2 = "#0d1829"
BG3 = "#111f35"
BORDER = "#1e3050"
CYAN = "#00e5ff"
GREEN = "#00e676"
RED = "#ff1744"
ORANGE = "#ff9100"
YELLOW = "#ffea00"
WHITE = "#e0f0ff"
GRAY = "#4a7090"

MURCOL = "#ff4040"
NORMCOL = "#00cc55"
EXTCOL = "#ffcc00"
ARTCOL = "#5577aa"

CLASS_COLOR = {"murmur": MURCOL, "normal": NORMCOL, "extrahls": EXTCOL, "artifact": ARTCOL}
BANNER_BG = {"normal": "#0a2a18", "murmur": "#2a0808", "extrahls": "#2a2000", "artifact": "#0a1520"}
BANNER_BORDER = {"normal": "#1a6a30", "murmur": "#aa2020", "extrahls": "#806600", "artifact": "#2a4a6a"}


def lbl(text: str, color: str, size: int, bold: bool = False, align=None) -> QLabel:
    widget = QLabel(text)
    weight = "bold" if bold else "normal"
    widget.setStyleSheet(f"color:{color};font-size:{size}px;font-weight:{weight};background:transparent;")
    if align:
        widget.setAlignment(align)
    return widget
