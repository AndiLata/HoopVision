"""Dashboard palette and text helpers.

All colours are BGR, because OpenCV. Swap TEXT_FONT for a PIL-drawn mono
TTF if you want the reel's exact terminal look - everything else here is
already sized for a monospaced face.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


@dataclass(frozen=True)
class Theme:
    bg: tuple = (238, 236, 233)
    panel: tuple = (255, 255, 255)
    ink: tuple = (24, 22, 19)
    ink_soft: tuple = (110, 104, 96)
    label: tuple = (168, 74, 28)        # the reel's blue labels
    rule: tuple = (216, 212, 206)
    grid: tuple = (232, 229, 225)
    accent: tuple = (39, 83, 188)       # burnt orange
    good: tuple = (86, 118, 52)
    bad: tuple = (52, 60, 152)
    warn: tuple = (40, 140, 200)
    ghost: tuple = (205, 200, 194)


THEME = Theme()


def panel(w: int, h: int, t: Theme = THEME) -> np.ndarray:
    img = np.full((h, w, 3), t.panel, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), t.rule, 1)
    return img


def label(img, text: str, x: int, y: int, t: Theme = THEME, scale: float = 1.0) -> None:
    """Small uppercase panel label, drawn in the accent-blue of the reel."""
    cv2.putText(img, text, (x, y), FONT_MONO, scale, t.label, 1, cv2.LINE_AA)


def value(img, text: str, x: int, y: int, t: Theme = THEME,
          scale: float = 1.6, color=None, thickness: int = 2) -> None:
    cv2.putText(img, text, (x, y), FONT, scale, color or t.ink, thickness, cv2.LINE_AA)


def small(img, text: str, x: int, y: int, t: Theme = THEME,
          scale: float = 0.9, color=None) -> None:
    cv2.putText(img, text, (x, y), FONT_MONO, scale, color or t.ink_soft, 1, cv2.LINE_AA)


def text_width(text: str, font=FONT, scale: float = 1.6, thickness: int = 2) -> int:
    (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    return w
