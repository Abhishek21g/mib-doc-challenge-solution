"""Render-first PDF ingestion with weak, positional text-layer signals."""

from __future__ import annotations

import hashlib
import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .models import CASE_ID_PATTERN


DEFAULT_RENDER_DPI = 200
MAX_RENDER_PIXELS = 12_000_000
MAX_TEXT_CHARS_PER_PAGE = 20_000


class RecoverableRenderError(RuntimeError):
    """A single PDF could not be rendered and may be omitted by BatchRunner."""


class RendererDependencyError(RuntimeError):
    """The image was built without a required pinned rendering dependency."""


@dataclass(frozen=True)
class Rect:
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "Rect":
        left, bottom, right, top = (float(value) for value in values)
        return cls(left=left, bottom=bottom, right=right, top=top)

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.top - self.bottom)

    def contains(self, other: "Rect", tolerance: float = 0.5) -> bool:
        return (
            other.left >= self.left - tolerance
            and other.bottom >= self.bottom - tolerance
            and other.right <= self.right + tolerance
            and other.top <= self.top + tolerance
        )

    def union(self, other: "Rect") -> "Rect":
        return Rect(
            left=min(self.left, other.left),
            bottom=min(self.bottom, other.bottom),
            right=max(self.right, other.right),
            top=max(self.top, other.top),
        )


@dataclass(frozen=True)
class TextSpan:
    page_index: int
    text: str
    box: Rect
    authoritative: bool = False
    off_crop: bool = False


@dataclass(frozen=True)
class RenderedPage:
    index: int
    image_png: bytes
    width_px: int
    height_px: int
    dpi: int
    rotation_deg: int
    skew_correction_deg: float
    crop_box: Rect
    text_spans: tuple[TextSpan, ...]
    text_layer_truncated: bool = False


@dataclass(frozen=True)
class RenderedCase:
    source_path: Path
    source_sha256: str
    case_id: str | None
    pages: tuple[RenderedPage, ...]
    text_layer: tuple[TextSpan, ...]


class TextLayerReader:
    """Read positional text as explicitly non-authoritative weak signals."""

    def __init__(self, *, max_chars_per_page: int = MAX_TEXT_CHARS_PER_PAGE) -> None:
        if max_chars_per_page < 1:
            raise ValueError("max_chars_per_page must be positive")
        self._max_chars_per_page = max_chars_per_page

    def read_page(
        self,
        page: Any,
        *,
        page_index: int,
        crop_box: Rect,
    ) -> tuple[tuple[TextSpan, ...], bool]:
        text_page = page.get_textpage()
        try:
            total_chars = int(text_page.count_chars())
            char_count = min(total_chars, self._max_chars_per_page)
            spans: list[TextSpan] = []
            text_parts: list[str] = []
            span_box: Rect | None = None
            span_off_crop = False

            def flush() -> None:
                nonlocal text_parts, span_box, span_off_crop
                text = "".join(text_parts).strip()
                if text and span_box is not None:
                    spans.append(
                        TextSpan(
                            page_index=page_index,
                            text=text,
                            box=span_box,
                            authoritative=False,
                            off_crop=span_off_crop,
                        )
                    )
                text_parts = []
                span_box = None
                span_off_crop = False

            for index in range(char_count):
                character = text_page.get_text_range(index, 1)
                if not character or character.isspace():
                    flush()
                    continue
                try:
                    char_box = Rect.from_values(text_page.get_charbox(index))
                except Exception:
                    flush()
                    continue
                text_parts.append(character)
                span_box = char_box if span_box is None else span_box.union(char_box)
                span_off_crop = span_off_crop or not crop_box.contains(char_box)
            flush()
            return tuple(spans), total_chars > char_count
        finally:
            text_page.close()


class DocumentRenderer:
    """Rasterize every page and retain text only as a weak side channel."""

    def __init__(
        self,
        *,
        target_dpi: int = DEFAULT_RENDER_DPI,
        max_render_pixels: int = MAX_RENDER_PIXELS,
        text_reader: TextLayerReader | None = None,
    ) -> None:
        if not 96 <= target_dpi <= 300:
            raise ValueError("target_dpi must be between 96 and 300")
        if max_render_pixels < 1_000_000:
            raise ValueError("max_render_pixels must be at least 1,000,000")
        self._target_dpi = target_dpi
        self._max_render_pixels = max_render_pixels
        self._text_reader = text_reader or TextLayerReader()
        # PDFium is process-safe but does not guarantee concurrent calls from
        # multiple threads in one process. BatchRunner may use four threads, so
        # each renderer instance serializes its PDFium boundary.
        self._render_lock = threading.Lock()

    @staticmethod
    def _dependencies() -> tuple[Any, Any, Any]:
        try:
            import numpy
            import pypdfium2
            from PIL import Image
        except ImportError as exc:
            raise RendererDependencyError(
                "pypdfium2, Pillow, and numpy must be installed from requirements.lock"
            ) from exc
        return pypdfium2, Image, numpy

    def _render_scale(self, width_points: float, height_points: float) -> float:
        requested_scale = self._target_dpi / 72.0
        requested_pixels = width_points * height_points * requested_scale**2
        if requested_pixels <= self._max_render_pixels:
            return requested_scale
        return math.sqrt(self._max_render_pixels / (width_points * height_points))

    @staticmethod
    def _estimate_skew(image: Any, image_module: Any, numpy_module: Any) -> float:
        grayscale = image.convert("L")
        grayscale.thumbnail((1200, 1200), image_module.Resampling.BILINEAR)
        pixels = numpy_module.asarray(grayscale)
        if pixels.size == 0 or int(numpy_module.count_nonzero(pixels < 210)) < 50:
            return 0.0

        def score(candidate: float) -> float:
            rotated = grayscale.rotate(
                candidate,
                resample=image_module.Resampling.BILINEAR,
                expand=False,
                fillcolor=255,
            )
            ink = numpy_module.asarray(rotated) < 210
            profile = ink.sum(axis=1).astype("float64")
            return float(numpy_module.var(profile))

        candidates = tuple(index / 2.0 for index in range(-6, 7))
        scores = {candidate: score(candidate) for candidate in candidates}
        best = max(candidates, key=lambda candidate: (scores[candidate], -abs(candidate)))
        base_score = scores[0.0]
        if abs(best) < 0.25 or scores[best] <= base_score * 1.05:
            return 0.0
        return best

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def render(self, pdf_path: Path) -> RenderedCase:
        with self._render_lock:
            return self._render_locked(Path(pdf_path))

    def _render_locked(self, pdf_path: Path) -> RenderedCase:
        pdf_path = Path(pdf_path)
        pdfium, image_module, numpy_module = self._dependencies()
        try:
            document = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            raise RecoverableRenderError(f"cannot open PDF: {pdf_path.name}") from exc

        pages: list[RenderedPage] = []
        all_spans: list[TextSpan] = []
        try:
            page_count = len(document)
            if page_count == 0:
                raise RecoverableRenderError(f"PDF has no pages: {pdf_path.name}")

            for page_index in range(page_count):
                page = document[page_index]
                try:
                    crop_box = Rect.from_values(page.get_cropbox())
                    rotation_deg = int(page.get_rotation()) % 360
                    text_spans, text_truncated = self._text_reader.read_page(
                        page,
                        page_index=page_index,
                        crop_box=crop_box,
                    )
                    width_points, height_points = page.get_size()
                    scale = self._render_scale(width_points, height_points)
                    bitmap = page.render(
                        scale=scale,
                        rotation=0,
                        fill_color=(255, 255, 255, 255),
                        draw_annots=True,
                        rev_byteorder=True,
                        optimize_mode="print",
                    )
                    try:
                        image = bitmap.to_pil().convert("RGB")
                    finally:
                        bitmap.close()

                    skew_correction = self._estimate_skew(
                        image, image_module, numpy_module
                    )
                    if skew_correction:
                        image = image.rotate(
                            skew_correction,
                            resample=image_module.Resampling.BICUBIC,
                            expand=True,
                            fillcolor="white",
                        )
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG", compress_level=1)
                    rendered_page = RenderedPage(
                        index=page_index,
                        image_png=buffer.getvalue(),
                        width_px=image.width,
                        height_px=image.height,
                        dpi=max(1, round(scale * 72)),
                        rotation_deg=rotation_deg,
                        skew_correction_deg=skew_correction,
                        crop_box=crop_box,
                        text_spans=text_spans,
                        text_layer_truncated=text_truncated,
                    )
                    pages.append(rendered_page)
                    all_spans.extend(text_spans)
                except RecoverableRenderError:
                    raise
                except Exception as exc:
                    raise RecoverableRenderError(
                        f"failed to render page {page_index + 1} of {pdf_path.name}"
                    ) from exc
                finally:
                    page.close()
        finally:
            document.close()

        filename_case_id = pdf_path.stem
        case_id = (
            filename_case_id
            if CASE_ID_PATTERN.fullmatch(filename_case_id)
            else None
        )
        return RenderedCase(
            source_path=pdf_path,
            source_sha256=self._sha256(pdf_path),
            case_id=case_id,
            pages=tuple(pages),
            text_layer=tuple(all_spans),
        )
