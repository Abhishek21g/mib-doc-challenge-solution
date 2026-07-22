"""Fail-closed RapidOCR (ONNX) recovery for UNKNOWN fee / missing risk flags.

Reimplements the strobl dual-OCR *idea* in our stack: primary Tesseract stays
authoritative; RapidOCR may only fill fields that are still unknown / empty,
and may only *add* non-none risk flags. Never invents approvals by itself.
"""

from __future__ import annotations

import io
import os
import threading
from typing import Any

import fitz
import numpy as np
from PIL import Image

_ENGINE = None
_LOCK = threading.Lock()


def _get_engine() -> Any | None:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _LOCK:
        if _ENGINE is not None:
            return _ENGINE
        try:
            from rapidocr import RapidOCR

            _ENGINE = RapidOCR()
        except Exception:
            _ENGINE = False  # type: ignore[assignment]
        return _ENGINE if _ENGINE is not False else None


def rapid_available() -> bool:
    if os.environ.get("MIB_NO_RAPID", "").strip() in {"1", "true", "yes"}:
        return False
    return _get_engine() is not None


def ocr_pixmap_text(pix: fitz.Pixmap) -> str:
    """Run RapidOCR on a PyMuPDF pixmap; return newline-joined text."""
    eng = _get_engine()
    if eng is None:
        return ""
    try:
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        arr = np.asarray(img)
        out = eng(arr)
    except Exception:
        return ""
    if out is None:
        return ""
    txts = getattr(out, "txts", None)
    if not txts:
        # Older rapidocr versions returned list-of-lists.
        if isinstance(out, (list, tuple)) and out:
            chunks: list[str] = []
            rows = out[0] if isinstance(out[0], (list, tuple)) else out
            for item in rows or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    chunks.append(str(item[1]))
            return "\n".join(chunks)
        return ""
    return "\n".join(str(t) for t in txts if t)


def ocr_page_text(page: fitz.Page, dpi: int = 200) -> str:
    """Rasterize a page and OCR with RapidOCR."""
    try:
        pix = page.get_pixmap(dpi=dpi)
    except Exception:
        return ""
    return ocr_pixmap_text(pix)


def ocr_page_fee_band(page: fitz.Page, dpi: int = 220) -> str:
    """OCR the upper ~40% band (fee receipt status / amount / waiver)."""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        w, h = img.size
        band = img.crop((0, 0, w, max(1, int(h * 0.42))))
        eng = _get_engine()
        if eng is None:
            return ""
        out = eng(np.asarray(band))
    except Exception:
        return ""
    if out is None:
        return ""
    txts = getattr(out, "txts", None)
    if not txts:
        return ""
    return "\n".join(str(t) for t in txts if t)
