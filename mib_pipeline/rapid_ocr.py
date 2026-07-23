"""Fail-closed RapidOCR (ONNX) + Tess fee-crop ensemble.

Reimplements the strobl dual-OCR *idea* in our stack: primary Tesseract stays
authoritative; RapidOCR / hi-res fee crops may only fill fields that are still
unknown / empty, and may only *add* non-none risk flags. Never invents
approvals by itself.
"""

from __future__ import annotations

import io
import os
import re
import threading
from typing import Any

import fitz
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

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


def _result_to_text(out: Any) -> str:
    if out is None:
        return ""
    txts = getattr(out, "txts", None)
    if txts:
        return "\n".join(str(t) for t in txts if t)
    if isinstance(out, (list, tuple)) and out:
        chunks: list[str] = []
        rows = out[0] if isinstance(out[0], (list, tuple)) else out
        for item in rows or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                chunks.append(str(item[1]))
        return "\n".join(chunks)
    return ""


def ocr_pixmap_text(pix: fitz.Pixmap) -> str:
    """Run RapidOCR on a PyMuPDF pixmap; return newline-joined text."""
    eng = _get_engine()
    if eng is None:
        return ""
    try:
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        out = eng(np.asarray(img))
    except Exception:
        return ""
    return _result_to_text(out)


def ocr_page_text(page: fitz.Page, dpi: int = 200) -> str:
    """Rasterize a page and OCR with RapidOCR."""
    try:
        pix = page.get_pixmap(dpi=dpi)
    except Exception:
        return ""
    return ocr_pixmap_text(pix)


def ocr_page_fee_band(page: fitz.Page, dpi: int = 220) -> str:
    """OCR the upper ~45% band (fee receipt status / amount / waiver)."""
    eng = _get_engine()
    if eng is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        w, h = img.size
        band = img.crop((0, 0, w, max(1, int(h * 0.45))))
        out = eng(np.asarray(band))
    except Exception:
        return ""
    return _result_to_text(out)


def ocr_page_oriented(page: fitz.Page, dpi: int = 180) -> str:
    """RapidOCR with cheap upright-first orientation (dw820-style)."""
    eng = _get_engine()
    if eng is None:
        return ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        rgb = np.asarray(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    except Exception:
        return ""

    def _run(arr: np.ndarray) -> str:
        try:
            return _result_to_text(eng(arr))
        except Exception:
            return ""

    best = _run(rgb)
    if len(re.findall(r"[A-Za-z]{3,}", best)) >= 8:
        return best
    best_score = len(best)
    for k in (1, 2, 3):
        rot = np.ascontiguousarray(np.rot90(rgb, k))
        txt = _run(rot)
        if len(txt) > best_score:
            best, best_score = txt, len(txt)
    return best


def tess_fee_crop_text(page: fitz.Page, dpi: int = 250) -> str:
    """Hi-res Tesseract fee-band ensemble (no Rapid). Fast path for UNKNOWN fees."""
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        gray = ImageOps.autocontrast(ImageOps.grayscale(img))
        gray = ImageEnhance.Sharpness(gray).enhance(2.0)
        w, h = gray.size
        bands = [
            gray.crop((0, 0, w, max(1, int(h * 0.55)))),
            gray.crop((0, 0, w, max(1, int(h * 0.40)))),
        ]
    except Exception:
        return ""
    chunks: list[str] = []
    for band in bands:
        for psm in ("6", "11"):
            try:
                txt = pytesseract.image_to_string(band, config=f"--psm {psm}")
            except Exception:
                continue
            if txt and txt.strip():
                chunks.append(txt)
    return "\n".join(chunks)


def page_looks_fee_candidate(page: fitz.Page) -> bool:
    """Sparse image page that may hold a fee receipt (labeled or mystery)."""
    native = page.get_text() or ""
    upper = native.upper()
    imgs = page.get_images()
    if not imgs:
        return "FEE RECEIPT" in upper or (
            "FEE STATUS" in upper and "AMOUNT" in upper
        )
    if "FEE" in upper or "RECEIPT" in upper or (
        "AMOUNT" in upper and "WAIVER" in upper
    ):
        return True
    if len(native) >= 150:
        return False
    # Mystery sparse image: exclude clearly other form types.
    blockers = (
        "BIOMETRIC",
        "B-13",
        "OBSERVED FLAGS",
        "REGISTRY",
        "INTAKE",
        "SPONSOR ATTESTATION",
        "PASSPORT",
        "FORM I-8090",
        "FORM I 8090",
        "ADJUDICATOR",
    )
    return not any(b in upper for b in blockers)


def page_looks_bio_candidate(page: fitz.Page) -> bool:
    native = page.get_text() or ""
    upper = native.upper()
    if "BIOMETRIC" in upper or "OBSERVED FLAGS" in upper or "B-13" in upper:
        return True
    if page.get_images() and len(native) < 120:
        blockers = (
            "FEE RECEIPT",
            "REGISTRY",
            "SPONSOR ATTESTATION",
            "FORM I-8090",
            "ADJUDICATOR",
        )
        return not any(b in upper for b in blockers)
    return False
