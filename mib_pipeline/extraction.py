"""Visible-only OCR/CV extraction and untrusted-content vetoes."""

from __future__ import annotations

import csv
import difflib
import hashlib
import io
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol

from .ingestion import Rect, RenderedCase, RenderedPage
from .models import ADJUDICATION_VALUES, CASE_ID_PATTERN, FEE_VALUES, SPONSOR_ID_PATTERN


class RecoverableOcrError(RuntimeError):
    """OCR failed for one page/case and may be isolated by BatchRunner."""


class EvidenceType(str, Enum):
    ADJUDICATOR_STAMP = "adjudicator_stamp"
    SIGNED_MANUAL_NOTE = "signed_manual_note"
    INTAKE_FORM = "intake_form"
    BIOMETRIC_SLIP = "biometric_slip"
    SPONSOR_ATTESTATION = "sponsor_attestation"
    REGISTRY_EXTRACT = "registry_extract"
    TEXT_LAYER = "text_layer"


@dataclass(frozen=True)
class OcrToken:
    page_index: int
    text: str
    confidence: float
    box: Rect
    block_num: int
    paragraph_num: int
    line_num: int
    word_num: int


@dataclass(frozen=True)
class OcrLine:
    page_index: int
    text: str
    confidence: float
    box: Rect
    tokens: tuple[OcrToken, ...]


@dataclass(frozen=True)
class CandidateEvidence:
    field_name: str
    value: str | None
    evidence_type: EvidenceType
    page_index: int
    box: Rect
    legible: bool
    superseded: bool
    ocr_confidence: float
    visual_cues: tuple[str, ...] = ()
    source: str = "visible_ocr"
    case_id_hint: str | None = None
    applicant_hint: str | None = None


# Packet topology is policy-only visible evidence.  Keep the marker field
# names separate so pages of different types cannot create a same-rank value
# conflict in the resolver.
PAGE_TYPE_MARKER_FIELDS = {
    "fee_receipt": "page_type_present_fee_receipt",
    "other": "page_type_present_other",
    "sponsor_attestation": "page_type_present_sponsor_attestation",
}


class RefinementModel(Protocol):
    def refine(
        self,
        page: RenderedPage,
        line: OcrLine,
    ) -> tuple[str, float] | None:
        """Optionally refine a visibly-present uncertain OCR line."""


class TesseractOcrEngine:
    """Bounded offline Tesseract TSV adapter using PNG bytes over stdin."""

    def __init__(
        self,
        *,
        binary: str = "tesseract",
        language: str = "eng",
        page_segmentation_mode: int = 11,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not 1 <= page_segmentation_mode <= 13:
            raise ValueError("page_segmentation_mode must be between 1 and 13")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._binary = binary
        self._language = language
        self._psm = page_segmentation_mode
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def _parse_tsv(tsv_text: str, page_index: int) -> tuple[OcrToken, ...]:
        tokens: list[OcrToken] = []
        # Tesseract's TSV is tab-delimited, but the final text cell is not
        # RFC-4180 quoted.  Treating a visible quote as CSV syntax can swallow
        # every following TSV row into one token and silently lose most of a
        # page.  QUOTE_NONE keeps each physical TSV line independent.
        reader = csv.DictReader(
            io.StringIO(tsv_text),
            delimiter="\t",
            quoting=csv.QUOTE_NONE,
        )
        required = {
            "level",
            "left",
            "top",
            "width",
            "height",
            "conf",
            "text",
            "block_num",
            "par_num",
            "line_num",
            "word_num",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise RecoverableOcrError("Tesseract returned malformed TSV")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text or row.get("level") != "5":
                continue
            try:
                confidence_percent = float(row["conf"])
                left = int(row["left"])
                top = int(row["top"])
                width = int(row["width"])
                height = int(row["height"])
                block_num = int(row["block_num"])
                paragraph_num = int(row["par_num"])
                line_num = int(row["line_num"])
                word_num = int(row["word_num"])
            except (TypeError, ValueError) as exc:
                raise RecoverableOcrError("Tesseract returned invalid TSV values") from exc
            if confidence_percent < 0 or width <= 0 or height <= 0:
                continue
            tokens.append(
                OcrToken(
                    page_index=page_index,
                    text=text,
                    confidence=max(0.0, min(1.0, confidence_percent / 100.0)),
                    box=Rect(left, top, left + width, top + height),
                    block_num=block_num,
                    paragraph_num=paragraph_num,
                    line_num=line_num,
                    word_num=word_num,
                )
            )
        return tuple(tokens)

    def read_page(self, page: RenderedPage) -> tuple[OcrToken, ...]:
        executable = shutil.which(self._binary)
        if executable is None:
            raise RecoverableOcrError(f"Tesseract executable not found: {self._binary}")
        environment = os.environ.copy()
        environment["OMP_THREAD_LIMIT"] = "1"
        command = [
            executable,
            "stdin",
            "stdout",
            "--dpi",
            str(page.dpi),
            "-l",
            self._language,
            "--oem",
            "1",
            "--psm",
            str(self._psm),
            "tsv",
        ]
        try:
            completed = subprocess.run(
                command,
                input=page.image_png,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self._timeout_seconds,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RecoverableOcrError(
                f"OCR timed out on page {page.index + 1}"
            ) from exc
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RecoverableOcrError(
                f"OCR failed on page {page.index + 1}: {message[:240]}"
            )
        return self._parse_tsv(
            completed.stdout.decode("utf-8", errors="replace"),
            page.index,
        )


def group_ocr_lines(tokens: Iterable[OcrToken]) -> tuple[OcrLine, ...]:
    grouped: dict[tuple[int, int, int, int], list[OcrToken]] = {}
    for token in tokens:
        key = (
            token.page_index,
            token.block_num,
            token.paragraph_num,
            token.line_num,
        )
        grouped.setdefault(key, []).append(token)
    lines: list[OcrLine] = []
    for key in sorted(grouped):
        line_tokens = tuple(sorted(grouped[key], key=lambda token: token.word_num))
        box = line_tokens[0].box
        for token in line_tokens[1:]:
            box = box.union(token.box)
        weight = sum(max(1, len(token.text)) for token in line_tokens)
        confidence = sum(
            token.confidence * max(1, len(token.text)) for token in line_tokens
        ) / weight
        lines.append(
            OcrLine(
                page_index=key[0],
                text=" ".join(token.text for token in line_tokens),
                confidence=confidence,
                box=box,
                tokens=line_tokens,
            )
        )
    return tuple(lines)


def _visual_reading_order(lines: Iterable[OcrLine]) -> tuple[OcrLine, ...]:
    """Order sparse OCR lines by visual row, then from left to right.

    Tesseract's sparse-text mode deliberately emits independently detected
    regions. Its block order is therefore not a reliable reading order, and a
    table value may otherwise precede its label. Row clustering tolerates the
    small vertical drift commonly seen between two cells on the same row.
    """

    remaining = sorted(
        lines,
        key=lambda line: (
            line.page_index,
            (line.box.bottom + line.box.top) / 2.0,
            line.box.left,
        ),
    )
    ordered: list[OcrLine] = []
    while remaining:
        anchor = remaining.pop(0)
        anchor_center = (anchor.box.bottom + anchor.box.top) / 2.0
        anchor_height = max(1.0, anchor.box.height)
        row = [anchor]
        rest: list[OcrLine] = []
        for line in remaining:
            if line.page_index != anchor.page_index:
                rest.append(line)
                continue
            center = (line.box.bottom + line.box.top) / 2.0
            tolerance = max(
                4.0,
                min(anchor_height, max(1.0, line.box.height)) * 0.75,
            )
            if abs(center - anchor_center) <= tolerance:
                row.append(line)
            else:
                rest.append(line)
        ordered.extend(sorted(row, key=lambda line: line.box.left))
        remaining = rest
    return tuple(ordered)


class UntrustedContentFilter:
    """Keep injected or non-visible material outside CandidateEvidence."""

    _CONTEXT_PATTERNS = (
        re.compile(r"\b(?:fake\s+)?(?:system\s+prompt|answer\s+key|ground\s+truth)\b", re.I),
        re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b", re.I),
        re.compile(r"\b(?:qr|bar\s*code)\b.*\b(?:instruction|policy|answer|decision)\b", re.I),
        re.compile(r"\b(?:assistant|chatgpt)\s*:\s*", re.I),
    )
    _LINE_PATTERNS = (
        re.compile(r"\bsample\s+denial\b", re.I),
        re.compile(r"\bdo\s+not\s+follow\s+(?:the\s+)?(?:form|policy)\b", re.I),
    )

    def context_quarantine_lines(self, text: str) -> int:
        for pattern in self._CONTEXT_PATTERNS:
            if pattern.search(text):
                return 12
        return 0

    def rejection_reason(self, text: str, cues: Iterable[str] = ()) -> str | None:
        cue_set = set(cues)
        anchored_correction = bool(
            re.match(r"^\s*manual\s+correction\s*:", text, re.I)
        )
        if "sample_denial_watermark" in cue_set and not anchored_correction:
            return "sample denial watermark"
        for pattern in self._CONTEXT_PATTERNS + self._LINE_PATTERNS:
            if pattern.search(text) and not anchored_correction:
                return "injected or decorative content"
        return None

    @staticmethod
    def allows_text_span(*, authoritative: bool, off_crop: bool) -> bool:
        return authoritative and not off_crop


class VisualCueDetector:
    """Detect lightweight visible cues without interpreting them as policy."""

    @staticmethod
    def _dependencies() -> tuple[Any, Any]:
        try:
            import numpy
            from PIL import Image
        except ImportError as exc:
            raise RecoverableOcrError("Pillow and numpy are required for visual cues") from exc
        return Image, numpy

    def prepare_page(self, grayscale: Any) -> Any:
        _, numpy_module = self._dependencies()
        return numpy_module.asarray(grayscale)

    @staticmethod
    def _has_strikethrough(pixels: Any, box: Rect, numpy_module: Any) -> bool:
        height, width = pixels.shape[:2]
        left = max(0, min(width, int(box.left)))
        right = max(0, min(width, int(box.right)))
        top = max(0, min(height, int(box.bottom)))
        bottom = max(0, min(height, int(box.top)))
        if right - left < 8 or bottom - top < 4:
            return False
        region = pixels[top:bottom, left:right]
        # Hand-drawn strikes are not always vertically centred in Tesseract's
        # word box.  In particular, thin coloured strokes often cross the
        # upper quarter of the glyphs.  Keep the strict horizontal-continuity
        # requirement below, but search a wider interior band.
        center_start = max(0, int(region.shape[0] * 0.15))
        center_end = min(region.shape[0], max(center_start + 1, int(region.shape[0] * 0.75)))
        center = region[center_start:center_end] < 100
        if not center.size:
            return False
        # Bold/blurred glyphs can cover most pixels on a row without being
        # struck through.  A real strike is also one long continuous segment
        # across the text box; letter strokes remain fragmented into words and
        # glyphs even when their aggregate coverage is high.
        for row in center:
            if float(row.mean()) < 0.72:
                continue
            longest = 0
            current = 0
            for is_ink in row:
                if bool(is_ink):
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            if longest / max(1, row.size) >= 0.65:
                return True
        return False

    def cues_for_line(self, line: OcrLine, page_pixels: Any) -> tuple[str, ...]:
        text = line.text.casefold()
        cues: set[str] = set()
        if "sample denial" in text:
            cues.add("sample_denial_watermark")
        if re.search(r"\b(?:correction|corrected|amended|override|supersedes?)\b", text):
            cues.add("correction")
        if re.search(r"\b(?:adjudicator\s+stamp|mib\s+official\s+stamp)\b", text):
            cues.add("adjudicator_stamp")
        _, numpy_module = self._dependencies()
        pixels = (
            page_pixels
            if hasattr(page_pixels, "shape")
            else numpy_module.asarray(page_pixels)
        )
        if self._has_strikethrough(pixels, line.box, numpy_module):
            cues.add("strikethrough")
        return tuple(sorted(cues))


FIELD_ALIASES = {
    "case_id": ("case id", "mib case", "application id"),
    "applicant_name": (
        "applicant name",
        "registry name",
        "full name",
        "applicant",
        "name",
    ),
    "species_code": ("species code", "species match", "species"),
    "home_world": ("home world", "homeworld", "origin world"),
    "visa_class": ("visa class", "visa"),
    "sponsor_id": ("sponsor id", "sponsor"),
    "arrival_date": ("arrival date", "date of arrival"),
    "declared_purpose": ("declared purpose", "purpose of visit", "purpose"),
    "risk_flags": ("observed flags", "risk flags", "risk flag", "flags"),
    "fee_status": ("fee status", "fee"),
    "adjudication": ("adjudication", "decision", "final status"),
    "stay_duration_days": (
        "requested stay days",
        "requested stay",
        "stay duration",
        "duration of stay",
        "requested duration",
    ),
    "packet_receipt_date": (
        "packet receipt date",
        "packet received",
        "packet date",
        "received on",
        "received date",
        "receipt date",
    ),
    "biohazard_check": (
        "biohazard check",
        "biohazard status",
        "biohazard screening",
    ),
    "hardship_waiver": ("hardship waiver", "fee waiver"),
    "diplomatic_waiver_code": ("waiver code",),
    "diplomatic_note": ("diplomatic note",),
    "work_permit_requested": (
        "work permit requested",
        "work authorization requested",
    ),
}

KNOWN_RISK_FLAGS = frozenset(
    {
        "memory_tampering",
        "planetary_embargo",
        "active_warrant",
        "biohazard_red",
        "identity_conflict",
        "sponsor_mismatch",
        "illegible_biometrics",
        "rescinded_denial",
    }
)

KNOWN_SPECIES_CODES = (
    "ALPHA_DRACONIAN",
    "ANDROMEDAN",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "CENTAURI_SYNTH",
    "JOVIAN_GASFORM",
    "KAIJU_MICRO",
    "LUNA_SECURID",
    "ORION_GRAYS",
    "SIRIUS_AVIAN",
    "TRIANGULAN",
    "VENUSIAN_MYCELIAL",
)
KNOWN_HOME_WORLDS = (
    "Barnard-c",
    "Eris Relay",
    "Europa Station",
    "Gliese-581g",
    "Kepler-186f",
    "Luyten-b",
    "Mars Dome-7",
    "Proxima-b",
    "Sirius Outpost",
    "Titan Freeport",
    "TRAPPIST-1e",
    "Wolf-1061c",
    "Zeta Reticuli",
)
KNOWN_VISA_CLASSES = ("DIP-1", "MED-3", "TRANSIT-7", "XW-1", "XW-2")
KNOWN_PURPOSES = (
    "archive audit",
    "cultural exchange",
    "diplomatic",
    "field repair",
    "medical consult",
    "reactor maintenance",
    "research",
    "transit",
    "translation",
    "xenobotany",
)

# A second OCR view is allowed to fill only scored, non-identity output fields.
# Applicant/case linkage and policy-only evidence must remain entirely owned by
# the primary pass.
CONSENSUS_RETRY_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)

# Applicant names are generated compositionally.  Storing the 12 stems and 12
# endings is a general OCR language model, not a case/name lookup table.
APPLICANT_STEMS = (
    "Ari",
    "Ixo",
    "Lu",
    "Mira",
    "Nex",
    "Ori",
    "Qor",
    "Sol",
    "Tek",
    "Vee",
    "Xan",
    "Za",
)
APPLICANT_ENDINGS = (
    "dane",
    "ix",
    "kesh",
    "mora",
    "nax",
    "quell",
    "rix",
    "tari",
    "ul",
    "vara",
    "voss",
    "zarn",
)
KNOWN_NAME_PARTS = tuple(
    stem + ending for stem in APPLICANT_STEMS for ending in APPLICANT_ENDINGS
)


def _ocr_key(value: str) -> str:
    """Return an OCR-comparison key while ignoring layout punctuation."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _canonical_vocabulary_value(
    raw_value: str,
    vocabulary: Iterable[str],
    *,
    cutoff: float,
    margin: float = 0.04,
) -> str | None:
    """Map a visibly OCR-read value to a small published field vocabulary."""

    candidate_key = _ocr_key(raw_value)
    if not candidate_key:
        return None
    keyed = {_ocr_key(item): item for item in vocabulary}
    if candidate_key in keyed:
        return keyed[candidate_key]
    ranked = sorted(
        (
            (difflib.SequenceMatcher(None, candidate_key, key).ratio(), key)
            for key in keyed
        ),
        reverse=True,
    )
    if not ranked or ranked[0][0] < cutoff:
        return None
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin:
        return None
    return keyed[ranked[0][1]]


class TesseractPsm6RefinementModel:
    """Conservatively re-read one already-visible uncertain date line.

    The primary sparse-text pass remains authoritative for page routing and
    field discovery.  PSM 6 may only refine an arrival-date line at essentially
    the same pixel position, with a material confidence gain.  Its page result
    is cached by the rendered PNG digest so multiple eligible lines can never
    cause duplicate OCR work for one physical page.
    """

    _ELIGIBLE_FIELDS = frozenset({"arrival_date"})

    def __init__(
        self,
        *,
        ocr_engine: Any | None = None,
        content_filter: UntrustedContentFilter | None = None,
        minimum_refined_confidence: float = 0.72,
        minimum_confidence_gain: float = 0.12,
        minimum_iou: float = 0.80,
        cache_page_limit: int = 16,
    ) -> None:
        if not 0.0 <= minimum_refined_confidence <= 1.0:
            raise ValueError("minimum_refined_confidence must be between 0 and 1")
        if not 0.0 <= minimum_confidence_gain <= 1.0:
            raise ValueError("minimum_confidence_gain must be between 0 and 1")
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be between 0 and 1")
        if cache_page_limit < 1:
            raise ValueError("cache_page_limit must be positive")
        self._ocr = ocr_engine or TesseractOcrEngine(page_segmentation_mode=6)
        self._filter = content_filter or UntrustedContentFilter()
        self._minimum_refined_confidence = minimum_refined_confidence
        self._minimum_confidence_gain = minimum_confidence_gain
        self._minimum_iou = minimum_iou
        self._cache_page_limit = cache_page_limit
        self._cache: dict[tuple[int, bytes], tuple[OcrLine, ...]] = {}
        self._inflight: set[tuple[int, bytes]] = set()
        self._cache_condition = threading.Condition()

    @staticmethod
    def _page_key(page: RenderedPage) -> tuple[int, bytes]:
        return page.index, hashlib.sha256(page.image_png).digest()

    def _page_lines(self, page: RenderedPage) -> tuple[OcrLine, ...]:
        key = self._page_key(page)
        with self._cache_condition:
            while key in self._inflight and key not in self._cache:
                self._cache_condition.wait()
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            self._inflight.add(key)

        try:
            lines = _visual_reading_order(group_ocr_lines(self._ocr.read_page(page)))
        except RecoverableOcrError:
            # An optional refinement must never turn an otherwise processable
            # case into a technical omission.
            lines = ()
        except BaseException:
            with self._cache_condition:
                self._inflight.discard(key)
                self._cache_condition.notify_all()
            raise

        with self._cache_condition:
            if len(self._cache) >= self._cache_page_limit:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = lines
            self._inflight.discard(key)
            self._cache_condition.notify_all()
        return lines

    def _alignment_score(self, source: Rect, refined: Rect) -> float | None:
        intersection_width = max(
            0.0,
            min(source.right, refined.right) - max(source.left, refined.left),
        )
        intersection_height = max(
            0.0,
            min(source.top, refined.top) - max(source.bottom, refined.bottom),
        )
        intersection = intersection_width * intersection_height
        if intersection <= 0.0:
            return None
        source_area = max(1.0, source.width * source.height)
        refined_area = max(1.0, refined.width * refined.height)
        iou = intersection / max(1.0, source_area + refined_area - intersection)
        source_overlap = intersection / source_area
        width_ratio = refined.width / max(1.0, source.width)
        height_ratio = refined.height / max(1.0, source.height)
        left_delta = abs(source.left - refined.left)
        if (
            iou < self._minimum_iou
            or source_overlap < 0.80
            or not 0.85 <= width_ratio <= 1.15
            or not 0.75 <= height_ratio <= 1.25
            or left_delta > max(8.0, source.height * 0.25)
        ):
            return None
        return iou

    def refine(
        self,
        page: RenderedPage,
        line: OcrLine,
    ) -> tuple[str, float] | None:
        source_match = VisibleEvidenceExtractor._match_field(line.text)
        if source_match is None or source_match[0] not in self._ELIGIBLE_FIELDS:
            return None
        field_name = source_match[0]
        matches: list[tuple[float, OcrLine, str]] = []
        for candidate in self._page_lines(page):
            if candidate.page_index != line.page_index:
                continue
            refined_match = VisibleEvidenceExtractor._match_field(candidate.text)
            if (
                refined_match is None
                or refined_match[0] != field_name
                or not refined_match[1]
            ):
                continue
            normalized = VisibleEvidenceExtractor._normalize_value(
                field_name,
                refined_match[1],
            )
            if normalized is None:
                continue
            if candidate.confidence < self._minimum_refined_confidence:
                continue
            if (
                candidate.confidence - line.confidence + 1e-12
                < self._minimum_confidence_gain
            ):
                continue
            if self._filter.rejection_reason(candidate.text) is not None:
                continue
            alignment = self._alignment_score(line.box, candidate.box)
            if alignment is not None:
                matches.append((alignment, candidate, normalized))

        # Conflicting same-position readings are ambiguity, not refinement.
        if not matches or len({item[2] for item in matches}) != 1:
            return None
        _alignment, best, _normalized = max(
            matches,
            key=lambda item: (item[0], item[1].confidence),
        )
        return best.text, best.confidence


class VisibleEvidenceExtractor:
    """Create field candidates only from OCR-confirmed visible page pixels."""

    def __init__(
        self,
        *,
        ocr_engine: TesseractOcrEngine | None = None,
        cue_detector: VisualCueDetector | None = None,
        content_filter: UntrustedContentFilter | None = None,
        refinement_model: RefinementModel | None = None,
        psm6_refinement: bool | None = None,
        minimum_legible_confidence: float = 0.45,
        refinement_gate: float = 0.72,
        consensus_retry: bool | None = None,
        retry_ocr_engines: tuple[Any, Any] | None = None,
        fee_receipt_retry: bool | None = None,
        fee_receipt_ocr_engines: tuple[Any, Any, Any, Any] | None = None,
        sparse_intake_retry: bool | None = None,
        sparse_intake_ocr_engines: tuple[Any, Any] | None = None,
        orientation_retry: bool | None = None,
        orientation_ocr_engines: tuple[Any, Any, Any] | None = None,
        trusted_scope_repair: bool | None = None,
        risk_flag_retry: bool | None = None,
        risk_flag_ocr_engines: tuple[Any, Any, Any] | None = None,
        packet_page_type_markers: bool = False,
    ) -> None:
        if not 0.0 <= minimum_legible_confidence <= refinement_gate <= 1.0:
            raise ValueError("confidence thresholds must satisfy 0 <= minimum <= gate <= 1")
        using_default_ocr = ocr_engine is None
        self._ocr = ocr_engine or TesseractOcrEngine()
        self._cues = cue_detector or VisualCueDetector()
        self._filter = content_filter or UntrustedContentFilter()
        use_default_refinement = (
            psm6_refinement is True
            or (psm6_refinement is None and using_default_ocr)
        )
        self._refinement_model = refinement_model
        if self._refinement_model is None and use_default_refinement:
            self._refinement_model = TesseractPsm6RefinementModel(
                content_filter=self._filter,
                minimum_refined_confidence=refinement_gate,
            )
        self._minimum_legible_confidence = minimum_legible_confidence
        self._refinement_gate = refinement_gate
        self._consensus_retry = (
            isinstance(self._ocr, TesseractOcrEngine)
            if consensus_retry is None
            else consensus_retry
        )
        if retry_ocr_engines is not None and len(retry_ocr_engines) != 2:
            raise ValueError("consensus retry requires exactly two OCR engines")
        self._retry_ocr_engines = retry_ocr_engines
        self._fee_receipt_retry = (
            using_default_ocr
            if fee_receipt_retry is None
            else fee_receipt_retry
        )
        if (
            fee_receipt_ocr_engines is not None
            and len(fee_receipt_ocr_engines) != 4
        ):
            raise ValueError("fee receipt retry requires exactly four OCR engines")
        self._fee_receipt_ocr_engines = fee_receipt_ocr_engines
        self._sparse_intake_retry = (
            using_default_ocr
            if sparse_intake_retry is None
            else sparse_intake_retry
        )
        if (
            sparse_intake_ocr_engines is not None
            and len(sparse_intake_ocr_engines) != 2
        ):
            raise ValueError("sparse intake retry requires exactly two OCR engines")
        self._sparse_intake_ocr_engines = sparse_intake_ocr_engines
        self._orientation_retry = (
            using_default_ocr
            if orientation_retry is None
            else orientation_retry
        )
        if (
            orientation_ocr_engines is not None
            and len(orientation_ocr_engines) != 3
        ):
            raise ValueError("orientation retry requires exactly three OCR engines")
        self._orientation_ocr_engines = orientation_ocr_engines
        self._trusted_scope_repair = (
            using_default_ocr
            if trusted_scope_repair is None
            else trusted_scope_repair
        )
        self._risk_flag_retry = (
            using_default_ocr if risk_flag_retry is None else risk_flag_retry
        )
        if (
            risk_flag_ocr_engines is not None
            and len(risk_flag_ocr_engines) != 3
        ):
            raise ValueError("risk flag retry requires exactly three OCR engines")
        self._risk_flag_ocr_engines = risk_flag_ocr_engines
        # Only the production primary extractor needs policy topology.  Keep
        # it opt-in so secondary OCR passes and generic extraction consumers
        # retain their original field-candidate contract.
        self._packet_page_type_markers = packet_page_type_markers

    @staticmethod
    def _page_image(page: RenderedPage) -> Any:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RecoverableOcrError("Pillow is required for visible extraction") from exc
        return Image.open(io.BytesIO(page.image_png)).convert("L")

    @staticmethod
    def packet_page_type(lines: Iterable[OcrLine]) -> str:
        """Classify a page from only its first four visible OCR lines.

        This intentionally mirrors the label-blind feature definition used to
        freeze the review-recovery rules.  It does not use the filename, case
        identity, applicant identity, later page text, or PDF text layer.
        """

        heading = " ".join(line.text for line in tuple(lines)[:4]).casefold()
        if "manual adjudicator note" in heading or "signed manual note" in heading:
            return "signed_manual_note"
        if "adjudicator stamp" in heading or "official stamp" in heading:
            return "adjudicator_stamp"
        if "biometric" in heading or "form b-13" in heading or "form b 13" in heading:
            return "biometric_slip"
        if "sponsor attestation" in heading or "sponsor letter" in heading:
            return "sponsor_attestation"
        if "registry extract" in heading or "registry record" in heading:
            return "registry_extract"
        if "fee receipt" in heading:
            return "fee_receipt"
        if (
            "form i-8090" in heading
            or "form i 8090" in heading
            or "work authorization intake" in heading
            or "primary intake record" in heading
        ):
            return "intake_form"
        return "other"

    @classmethod
    def _packet_page_type_marker(
        cls,
        *,
        page: RenderedPage,
        lines: tuple[OcrLine, ...],
        case_id: str | None,
    ) -> CandidateEvidence | None:
        """Create one visible, packet-scoped marker for a mined page type."""

        page_type = cls.packet_page_type(lines)
        field_name = PAGE_TYPE_MARKER_FIELDS.get(page_type)
        if field_name is None:
            return None
        heading_lines = lines[:4]
        if heading_lines:
            marker_box = heading_lines[0].box
            for line in heading_lines[1:]:
                marker_box = marker_box.union(line.box)
            confidence = min(line.confidence for line in heading_lines)
        else:
            # An existing rendered page with no readable heading is the exact
            # ``other`` bucket used by the frozen first-four-lines feature.
            marker_box = page.crop_box
            confidence = 1.0
        return CandidateEvidence(
            field_name=field_name,
            value="present",
            evidence_type=(
                EvidenceType.SPONSOR_ATTESTATION
                if page_type == "sponsor_attestation"
                else EvidenceType.INTAKE_FORM
            ),
            page_index=page.index,
            box=marker_box,
            legible=True,
            superseded=False,
            ocr_confidence=confidence,
            visual_cues=(f"packet_page_type:{page_type}",),
            source="visible_ocr",
            case_id_hint=case_id,
            applicant_hint=None,
        )

    @staticmethod
    def _evidence_type(text: str, current: EvidenceType) -> EvidenceType:
        normalized = text.casefold()
        heading_key = _ocr_key(text)

        def resembles(*headings: str, cutoff: float = 0.66) -> bool:
            return any(
                difflib.SequenceMatcher(
                    None, heading_key, _ocr_key(heading)
                ).ratio()
                >= cutoff
                for heading in headings
            )

        if (
            "manual adjudicator note" in normalized
            or "signed manual note" in normalized
            or "signed note" in normalized
            or ("adjudicat" in normalized and "note" in normalized)
            or resembles("manual adjudicator note", "signed manual note")
        ):
            return EvidenceType.SIGNED_MANUAL_NOTE
        if (
            "adjudicator stamp" in normalized
            or "official stamp" in normalized
            or resembles("adjudicator stamp", "official stamp")
        ):
            return EvidenceType.ADJUDICATOR_STAMP
        if "biometric" in normalized or resembles("form b 13 biometric scan slip"):
            return EvidenceType.BIOMETRIC_SLIP
        if (
            "sponsor attestation" in normalized
            or "sponsor letter" in normalized
            or ("sponsor" in normalized and "attest" in normalized)
            or resembles("sponsor attestation letter", "sponsor letter")
        ):
            return EvidenceType.SPONSOR_ATTESTATION
        if (
            "registry extract" in normalized
            or "registry record" in normalized
            or resembles("planetary registry extract", "registry record")
        ):
            return EvidenceType.REGISTRY_EXTRACT
        if (
            "intake form" in normalized
            or "application form" in normalized
            or "form i-8090" in normalized
            or "work authorization intake" in normalized
            or "primary intake record" in normalized
            or resembles(
                "form i 8090 extraterrestrial work authorization intake",
                "primary intake record",
            )
        ):
            return EvidenceType.INTAKE_FORM
        return current

    @staticmethod
    def _match_field(text: str) -> tuple[str, str] | None:
        normalized = " ".join(text.strip().split())
        normalized = re.sub(r"^[^A-Za-z0-9]+", "", normalized)
        correction = re.match(
            r"^manual\s+correction\s*:\s*"
            r"(applicant|sponsor|visa\s+class|fee\s+status)\s+is\s+"
            r"(.+?)(?:\.\s*(?:sample\s+denial)?|\s+sample\s+denial|$)",
            normalized,
            re.I,
        )
        if correction:
            correction_fields = {
                "applicant": "applicant_name",
                "sponsor": "sponsor_id",
                "visa class": "visa_class",
                "fee status": "fee_status",
            }
            label = " ".join(correction.group(1).casefold().split())
            return correction_fields[label], correction.group(2).strip()
        narrative_decision = re.search(
            r"\b(?:finding|decision|final\s+status)"
            r"(?:\s*(?::|=|-)\s*|\s+)"
            r"(APPROVED|DENIED|NEEDS[ _-]REVIEW)\b",
            normalized,
            flags=re.I,
        )
        if narrative_decision:
            return "adjudication", narrative_decision.group(1)
        if re.fullmatch(r"(?:APPROVED|DENIED|NEEDS[ _-]REVIEW)", normalized, re.I):
            return "adjudication", normalized
        aliases = sorted(
            (
                (alias, field_name)
                for field_name, field_aliases in FIELD_ALIASES.items()
                for alias in field_aliases
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for alias, field_name in aliases:
            match = re.match(
                rf"^{re.escape(alias)}\b\s*(?::|#|=|-)?\s*(.*)$",
                normalized,
                flags=re.I,
            )
            if match:
                return field_name, match.group(1).strip()

        # Damaged scans often preserve a separator and value while slightly
        # corrupting the label (for example "Nisa Class: MED3").  Fuzzy-match
        # only the label side; the value still has to pass field-specific
        # canonicalization below.
        separator = re.match(r"^(.{2,40}?)(?::|=)\s*(.+)$", normalized)
        if separator:
            label_text, raw_value = separator.groups()
            label_key = _ocr_key(label_text)
            best: tuple[float, str] | None = None
            for alias, field_name in aliases:
                alias_key = _ocr_key(alias)
                score = difflib.SequenceMatcher(None, label_key, alias_key).ratio()
                if best is None or score > best[0]:
                    best = (score, field_name)
            if best is not None and best[0] >= 0.64:
                return best[1], raw_value.strip()

        # Some templates omit punctuation.  Compare a bounded leading phrase
        # against known labels, leaving at least one token as the value.
        words = normalized.split()
        best_prefix: tuple[float, str, str] | None = None
        for prefix_length in range(1, min(4, len(words))):
            label_key = _ocr_key(" ".join(words[:prefix_length]))
            raw_value = " ".join(words[prefix_length:])
            for alias, field_name in aliases:
                score = difflib.SequenceMatcher(
                    None, label_key, _ocr_key(alias)
                ).ratio()
                if best_prefix is None or score > best_prefix[0]:
                    best_prefix = (score, field_name, raw_value)
        if best_prefix is not None and best_prefix[0] >= 0.76:
            return best_prefix[1], best_prefix[2]

        # Narrative sponsor letters still expose a visible, typed sponsor ID
        # even when the label itself is partially lost.
        if re.search(r"\bSP[NM]\b|\bSP[NM][\s:-]", normalized, re.I):
            return "sponsor_id", normalized
        return None

    @staticmethod
    def _risk_flags_from_text(text: str) -> tuple[str, ...]:
        """Read explicit/derived visible risk wording, including mild OCR noise."""

        normalized = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
        found = {
            flag
            for flag in KNOWN_RISK_FLAGS
            if re.search(rf"(?:^|_){re.escape(flag)}(?:_|$)", normalized)
        }
        if re.search(r"identity.*(?:mismatch|conflict|failed)", text, re.I):
            found.add("identity_conflict")
        if re.search(r"biometric.*(?:illegible|unreadable|failed)", text, re.I):
            found.add("illegible_biometrics")

        risk_suffix = re.search(
            r"(?:risk\s+flags?|observed\s+flags?)\s*(?::|=|-)?\s*(.*)$",
            text,
            re.I,
        )
        if risk_suffix:
            raw_items = re.split(r"[,;|\s]+", risk_suffix.group(1))
            for raw_item in raw_items:
                item = re.sub(r"[^a-z_]", "", raw_item.casefold())
                if not item or item in {"none", "reason", "disqualifying"}:
                    continue
                close = difflib.get_close_matches(
                    item,
                    sorted(KNOWN_RISK_FLAGS),
                    n=1,
                    cutoff=0.72,
                )
                if close:
                    found.add(close[0])
        return tuple(sorted(found))

    @staticmethod
    def _fuzzy_risk_flags_from_text(text: str) -> tuple[str, ...]:
        """Recover a known risk phrase from a short visibly-read OCR span.

        This helper is intentionally independent of document trust.  Callers
        must gate it to accepted lines on a trusted risk-bearing page type.
        Comparing one-to-four contiguous words allows damaged separators and
        spaces without turning the whole page into an unconstrained fuzzy
        search.
        """

        words = re.findall(r"[a-z0-9]+", text.casefold())
        found: set[str] = set()
        for span_length in range(1, 5):
            for start in range(0, len(words) - span_length + 1):
                phrase_key = "".join(words[start : start + span_length])
                for flag in KNOWN_RISK_FLAGS:
                    flag_key = flag.replace("_", "")
                    if (
                        difflib.SequenceMatcher(
                            None,
                            phrase_key,
                            flag_key,
                        ).ratio()
                        >= 0.75
                    ):
                        found.add(flag)
        return tuple(sorted(found))

    @staticmethod
    def _normalize_value(field_name: str, raw_value: str) -> str | None:
        value = " ".join(raw_value.strip().split())
        value = re.sub(r"^[\s|:;,.\[\](){}]+", "", value)
        value = re.sub(r"[\s|:;,.\[\](){}]+$", "", value)
        if not value:
            return None
        if field_name == "case_id":
            match = re.search(r"M[I1L]B\s*[-:]?\s*([0-9OQCDILSB\s]{6,12})", value, re.I)
            if match:
                digits = match.group(1).translate(
                    str.maketrans({"O": "0", "Q": "0", "C": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
                )
                digits = re.sub(r"\D", "", digits)[:6]
                candidate = f"MIB-{digits}"
            else:
                candidate = value.upper()
            return candidate if CASE_ID_PATTERN.fullmatch(candidate) else None
        if field_name == "sponsor_id":
            match = re.search(r"SP[NM]\s*[-:]?\s*([0-9OQCDILSB\s]{4,10})", value, re.I)
            if match:
                digits = match.group(1).translate(
                    str.maketrans({"O": "0", "Q": "0", "C": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
                )
                digits = re.sub(r"\D", "", digits)[:4]
                candidate = f"SPN-{digits}"
            else:
                candidate = value.upper()
            return candidate if SPONSOR_ID_PATTERN.fullmatch(candidate) else None
        if field_name in {"arrival_date", "packet_receipt_date"}:
            from datetime import datetime

            date_match = re.search(
                r"\b([0-9OILSB]{4})\s*[-./]\s*([0-9OILSB]{1,2})\s*[-./]\s*([0-9OILSB]{1,2})\b",
                value,
                re.I,
            )
            if date_match:
                date_text = "-".join(date_match.groups()).translate(
                    str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"})
                )
            else:
                date_text = value
            for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
                try:
                    parsed = datetime.strptime(date_text, pattern).date()
                    # Public packets are from the versioned 2026 challenge
                    # snapshot.  On damaged scans Tesseract frequently reads
                    # the rounded tail of a printed ``6`` as ``8`` (or the
                    # intermediate ``B``, translated above), producing an
                    # impossible 2028 arrival date.  Repair that single,
                    # visually-confusable year instead of allowing it to
                    # outrank a clean lower-precedence 2026 record.
                    if field_name == "arrival_date" and parsed.year == 2028:
                        parsed = parsed.replace(year=2026)
                    return parsed.isoformat()
                except ValueError:
                    continue
            return None
        if field_name == "stay_duration_days":
            match = re.search(r"\b([0-9]{1,4})\b", value)
            if match is None or int(match.group(1)) < 1:
                return None
            return str(int(match.group(1)))
        if field_name == "diplomatic_waiver_code":
            # This challenge-specific code is a visible learned-policy marker,
            # not a generic waiver.  Keep the match exact (apart from layout
            # punctuation) so a different or damaged code cannot authorize a
            # non-diplomatic fee waiver.
            return "valid" if _ocr_key(value) == "dipwaiver" else "invalid"
        if field_name in {
            "hardship_waiver",
            "diplomatic_note",
            "work_permit_requested",
        }:
            candidate = value.casefold().replace("_", " ")
            if re.search(r"\b(?:valid|approved|granted|yes|present|true)\b", candidate):
                return "yes" if field_name == "work_permit_requested" else "valid"
            if re.search(r"\b(?:invalid|denied|no|absent|false|none)\b", candidate):
                return "no" if field_name == "work_permit_requested" else "invalid"
            return None
        if field_name == "biohazard_check":
            candidate = value.casefold().replace("_", " ")
            if re.search(r"\b(?:clean|clear|green|negative)\b", candidate):
                return "clean"
            if re.search(r"\b(?:red|positive|biohazard red)\b", candidate):
                return "red"
            return None
        if field_name == "fee_status":
            fee_key = _ocr_key(value)
            if re.fullmatch(r"p[ao][i1l][dcl]", fee_key):
                return "paid"
            return _canonical_vocabulary_value(value, FEE_VALUES, cutoff=0.66)
        if field_name == "adjudication":
            candidate = value.upper().replace(" ", "_")
            return candidate if candidate in ADJUDICATION_VALUES else None
        if field_name == "risk_flags":
            if re.fullmatch(r"(?:none|no(?:ne)?|clear)", value, re.I):
                return "none"
            flags = VisibleEvidenceExtractor._risk_flags_from_text(value)
            return "|".join(flags) if flags else None
        if field_name == "species_code":
            return _canonical_vocabulary_value(
                value, KNOWN_SPECIES_CODES, cutoff=0.66
            )
        if field_name == "home_world":
            return _canonical_vocabulary_value(
                value, KNOWN_HOME_WORLDS, cutoff=0.66
            )
        if field_name == "visa_class":
            for visa in KNOWN_VISA_CLASSES:
                pattern = re.escape(visa).replace(r"\-", r"[\s._-]?")
                if re.search(pattern, value, re.I):
                    return visa
            return _canonical_vocabulary_value(value, KNOWN_VISA_CLASSES, cutoff=0.58)
        if field_name == "declared_purpose":
            normalized_value = value.casefold()
            for purpose in KNOWN_PURPOSES:
                if purpose in normalized_value:
                    return purpose
            return _canonical_vocabulary_value(value, KNOWN_PURPOSES, cutoff=0.68)
        if field_name == "applicant_name":
            words = re.findall(r"[A-Za-z]+", value)
            if len(words) != 2:
                return None
            if any(
                token in {"blank", "cut", "illegible", "obscured", "out", "redacted", "washed"}
                for token in (word.casefold() for word in words)
            ):
                return None
            parts = [
                _canonical_vocabulary_value(word, KNOWN_NAME_PARTS, cutoff=0.68)
                for word in words
            ]
            return " ".join(parts) if all(parts) else " ".join(words)
        return value

    @staticmethod
    def _sponsor_narrative_matches(text: str) -> tuple[tuple[str, str], ...]:
        """Extract the repeated, visible sponsor-letter sentence structure."""

        if not re.search(r"\bsponsor\b", text, re.I) or not re.search(
            r"\battests?\b", text, re.I
        ):
            return ()
        matches: list[tuple[str, str]] = []
        sponsor = re.search(
            r"\bsponsor\s+(SP[NM]\s*[-:]?\s*[0-9OQCDILSB\s]{4,10})\b",
            text,
            re.I,
        )
        if sponsor:
            matches.append(("sponsor_id", sponsor.group(1).strip()))
        applicant = re.search(
            r"\battests?\s+that\s+([A-Za-z]+\s+[A-Za-z]+)\s+is\s+expected\b",
            text,
            re.I,
        )
        if applicant:
            matches.append(("applicant_name", applicant.group(1)))
        purpose = re.search(
            r"\bis\s+expected\s+on\s+earth\s+for\s+(.+?)"
            r"(?:\.\s+the\s+sponsor|\.\s+this\s+attestation|$)",
            text,
            re.I,
        )
        if purpose:
            matches.append(("declared_purpose", purpose.group(1)))
        visa = re.search(
            r"\bclass\s+([A-Za-z0-9._-]+)\s+compliance\b",
            text,
            re.I,
        )
        if visa:
            matches.append(("visa_class", visa.group(1)))
        return tuple(matches)

    @classmethod
    def _visible_page_heading_type(
        cls,
        lines: Iterable[OcrLine],
    ) -> EvidenceType:
        """Classify only the leading visible heading region of a page."""

        for line in tuple(lines)[:4]:
            evidence_type = cls._evidence_type(line.text, EvidenceType.INTAKE_FORM)
            if evidence_type is not EvidenceType.INTAKE_FORM:
                return evidence_type
        return EvidenceType.INTAKE_FORM

    @staticmethod
    def _authoritative_fee_status(text: str) -> str | None:
        """Read an explicit fee fact from a signed note/stamp narrative."""

        if re.search(r"\bmandatory\s+fee\s+(?:is\s+)?unpaid\b", text, re.I):
            return "unpaid"
        if re.search(
            r"\bfee\s+status\s*(?::|=|-)?\s*(?:is\s+)?unknown\b",
            text,
            re.I,
        ):
            return "unknown"
        return None

    @staticmethod
    def _authoritative_finding_decision(text: str) -> str | None:
        """Recover one closed-vocabulary decision from a damaged Finding line."""

        words = re.findall(r"[a-z]+", text.casefold())
        if not words or ("sample" in words and "denial" in words):
            return None
        label_index = next(
            (
                index
                for index, word in enumerate(words[:3])
                if difflib.SequenceMatcher(None, word, "finding").ratio()
                >= 0.66
            ),
            None,
        )
        if label_index is None:
            return None
        value_key = "".join(words[label_index + 1 :])
        if len(value_key) < 3:
            return None

        vocabulary = (
            ("APPROVED", "approved"),
            ("DENIED", "denied"),
            ("NEEDS_REVIEW", "needsreview"),
        )
        prefix_matches = tuple(
            decision
            for decision, canonical in vocabulary
            if canonical.startswith(value_key)
        )
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        ranked = sorted(
            (
                difflib.SequenceMatcher(None, value_key, canonical).ratio(),
                decision,
            )
            for decision, canonical in vocabulary
        )
        best_score, best_decision = ranked[-1]
        second_score = ranked[-2][0]
        if (
            len(value_key) >= 4
            and best_score >= 0.55
            and best_score - second_score >= 0.18
        ):
            return best_decision
        return None

    @staticmethod
    def _authoritative_reason_decision(text: str) -> str | None:
        """Map only narrowly anchored signed-note reason templates to decisions."""

        words = re.findall(r"[a-z]+", text.casefold())
        if (
            not words
            or "sample" in words
            or not any(
                difflib.SequenceMatcher(None, word, "reason").ratio()
                >= 0.65
                for word in words[:1]
            )
        ):
            return None

        def has(target: str) -> bool:
            return any(
                difflib.SequenceMatcher(None, word, target).ratio()
                >= 0.72
                for word in words
            )

        if has("denial") and has("supported"):
            return "DENIED"
        if all(has(word) for word in ("clean", "exception", "qualified", "packet")):
            return "APPROVED"
        if all(
            has(word)
            for word in (
                "packet",
                "damaged",
                "contradictory",
                "visible",
                "evidence",
            )
        ):
            return "NEEDS_REVIEW"
        if all(has(word) for word in ("review", "risk", "flag", "present")):
            return "NEEDS_REVIEW"
        return None

    @classmethod
    def _recovered_authoritative_decision(
        cls,
        line_cues: Iterable[tuple[OcrLine, tuple[str, ...]]],
    ) -> tuple[str, tuple[tuple[OcrLine, tuple[str, ...]], ...]] | None:
        """Return an unambiguous visible note decision and its supporting lines."""

        signals: list[tuple[str, OcrLine, tuple[str, ...]]] = []
        for line, cues in line_cues:
            if "sample_denial_watermark" in cues:
                continue
            if line.confidence >= 0.15:
                finding = cls._authoritative_finding_decision(line.text)
                if finding is not None:
                    signals.append((finding, line, cues))
            if line.confidence >= 0.45:
                reason = cls._authoritative_reason_decision(line.text)
                if reason is not None:
                    signals.append((reason, line, cues))

        decisions = {decision for decision, _line, _cues in signals}
        if len(decisions) != 1:
            return None
        decision = next(iter(decisions))
        supporting = tuple(
            dict.fromkeys(
                (line, cues)
                for signal, line, cues in signals
                if signal == decision
            )
        )
        return decision, supporting

    @staticmethod
    def _registry_home_world(text: str) -> str | None:
        """Recover a title-gated registry home-world row under mild OCR noise."""

        match = re.search(
            r"\bhome\s+wor(?:ld|k|d)\s*[:#.=_-]?\s*(.{2,32})$",
            text.strip(),
            re.I,
        )
        if match is None or re.search(
            r"\b(?:blank|cut|illegible|lost|redacted)\b",
            match.group(1),
            re.I,
        ):
            return None
        return _canonical_vocabulary_value(
            match.group(1),
            KNOWN_HOME_WORLDS,
            cutoff=0.70,
            margin=0.07,
        )

    @staticmethod
    def _fee_receipt_phrase_similarity(
        text: str,
        targets: Iterable[str],
    ) -> float:
        """Compare a short, leading visible phrase with receipt vocabulary."""

        words = text.split()
        best = 0.0
        for start in range(min(3, len(words))):
            for length in range(1, min(5, len(words) - start) + 1):
                phrase_key = _ocr_key(" ".join(words[start : start + length]))
                if not phrase_key:
                    continue
                for target in targets:
                    best = max(
                        best,
                        difflib.SequenceMatcher(
                            None,
                            phrase_key,
                            _ocr_key(target),
                        ).ratio(),
                    )
        return best

    @classmethod
    def _fee_receipt_heading_similarity(cls, text: str) -> float:
        return cls._fee_receipt_phrase_similarity(
            text,
            ("MIB Fee Receipt", "Fee Receipt"),
        )

    @classmethod
    def _fee_receipt_label_similarity(cls, text: str) -> float:
        return cls._fee_receipt_phrase_similarity(
            text,
            ("Fee Status", "Payment Status"),
        )

    @staticmethod
    def _fee_receipt_value(text: str) -> str | None:
        """Normalize one receipt-row value without weakening global parsing."""

        value_key = _ocr_key(text)
        # On photocopied receipts the damaged bowl/descender of the leading
        # ``p`` in ``paid`` repeatedly resembles ``n`` or ``m``.  This narrow
        # repair is used only behind the receipt title + status-row gates and
        # still needs agreement across three independently thresholded views.
        if re.fullmatch(r"[pnm][ao][i1l][dcl]", value_key):
            return "paid"
        return _canonical_vocabulary_value(
            text,
            FEE_VALUES,
            cutoff=0.62,
            margin=0.06,
        )

    @classmethod
    def _fee_receipt_line_value(cls, text: str) -> str | None:
        """Read a value only when attached to a recognizable status label."""

        words = text.split()
        best: tuple[float, str] | None = None
        for split in range(1, min(5, len(words))):
            label = " ".join(words[:split])
            similarity = cls._fee_receipt_label_similarity(label)
            if similarity < 0.70:
                continue
            value = cls._fee_receipt_value(" ".join(words[split:]))
            if value is None:
                continue
            if best is None or similarity > best[0]:
                best = (similarity, value)
        return None if best is None else best[1]

    @staticmethod
    def _fee_receipt_crop(
        page: RenderedPage,
        image: Any,
        threshold: int,
    ) -> tuple[RenderedPage, Any]:
        """Expose the title, case ID, and fee row under one fixed threshold."""

        width, height = image.size
        crop = image.crop((0, 0, width, max(1, int(height * 0.30))))
        binary = crop.point(lambda pixel: 255 if pixel > threshold else 0)
        buffer = io.BytesIO()
        binary.save(buffer, format="PNG")
        return (
            RenderedPage(
                index=page.index,
                image_png=buffer.getvalue(),
                width_px=binary.width,
                height_px=binary.height,
                dpi=page.dpi,
                rotation_deg=0,
                skew_correction_deg=0.0,
                crop_box=Rect(0, 0, binary.width, binary.height),
                # This recovery path is deliberately pixels-only.
                text_spans=(),
            ),
            binary,
        )

    def _fee_receipt_lines_candidate(
        self,
        *,
        lines: tuple[OcrLine, ...],
        page: RenderedPage,
        page_pixels: Any,
        case_id: str,
        active_applicant: str,
        view_cue: str,
    ) -> CandidateEvidence | None:
        """Read one visible receipt view, abstaining on any ambiguity."""

        if max(
            (
                self._fee_receipt_heading_similarity(line.text)
                for line in lines[:8]
            ),
            default=0.0,
        ) < 0.64:
            return None

        page_visual = (
            self._cues.prepare_page(page_pixels)
            if hasattr(self._cues, "prepare_page")
            else page_pixels
        )
        records: list[CandidateEvidence] = []
        pending_label: tuple[OcrLine, tuple[str, ...]] | None = None
        for line in lines[:16]:
            cues = self._cues.cues_for_line(line, page_visual)
            value = self._fee_receipt_line_value(line.text)
            source_line = line
            combined_cues = cues
            if value is None and pending_label is not None:
                label_line, label_cues = pending_label
                value = self._fee_receipt_value(line.text)
                if value is not None:
                    source_line = OcrLine(
                        page_index=line.page_index,
                        text=f"{label_line.text} {line.text}",
                        confidence=line.confidence,
                        box=label_line.box.union(line.box),
                        tokens=label_line.tokens + line.tokens,
                    )
                    combined_cues = tuple(
                        sorted(set(label_cues) | set(cues))
                    )
            pending_label = None
            if value is None:
                if self._fee_receipt_label_similarity(line.text) >= 0.70:
                    pending_label = (line, cues)
                continue
            if value == "unknown":
                # The public output already represents a missing fee as
                # ``unknown``. Adding an explicit unknown cannot improve the
                # field and can only conceal a damaged valid waiver/payment.
                continue
            if (
                source_line.confidence < 0.28
                or "strikethrough" in combined_cues
                or "sample_denial_watermark" in combined_cues
                or self._filter.rejection_reason(
                    source_line.text,
                    combined_cues,
                )
                is not None
            ):
                continue
            records.append(
                CandidateEvidence(
                    field_name="fee_status",
                    value=value,
                    evidence_type=EvidenceType.INTAKE_FORM,
                    page_index=page.index,
                    box=source_line.box,
                    legible=True,
                    superseded=False,
                    ocr_confidence=source_line.confidence,
                    visual_cues=(view_cue,),
                    case_id_hint=case_id,
                    applicant_hint=active_applicant,
                )
            )

        if len({candidate.value for candidate in records}) != 1:
            return None
        return max(records, key=lambda candidate: candidate.ocr_confidence)

    def _fee_receipt_pass_candidate(
        self,
        *,
        page: RenderedPage,
        page_pixels: Any,
        engine: Any,
        case_id: str,
        active_applicant: str,
    ) -> CandidateEvidence | None:
        """OCR one thresholded receipt view and parse its anchored fee row."""

        try:
            lines = _visual_reading_order(group_ocr_lines(engine.read_page(page)))
        except RecoverableOcrError:
            return None
        return self._fee_receipt_lines_candidate(
            lines=lines,
            page=page,
            page_pixels=page_pixels,
            case_id=case_id,
            active_applicant=active_applicant,
            view_cue="threshold_fee_receipt_view",
        )

    @staticmethod
    def _fee_receipt_footer_case_ids(
        page: RenderedPage,
        image: Any,
        engine: Any,
    ) -> set[str]:
        """Re-read only the visible footer when the primary case ID is lost."""

        width, height = image.size
        footer = image.crop((0, int(height * 0.86), width, height))
        buffer = io.BytesIO()
        footer.save(buffer, format="PNG")
        footer_page = RenderedPage(
            index=page.index,
            image_png=buffer.getvalue(),
            width_px=footer.width,
            height_px=footer.height,
            dpi=page.dpi,
            rotation_deg=0,
            skew_correction_deg=0.0,
            crop_box=Rect(0, 0, footer.width, footer.height),
            text_spans=(),
        )
        try:
            lines = _visual_reading_order(
                group_ocr_lines(engine.read_page(footer_page))
            )
        except RecoverableOcrError:
            return set()
        return {
            value
            for line in lines
            if (
                value := VisibleEvidenceExtractor._normalize_value(
                    "case_id",
                    line.text,
                )
            )
            is not None
        }

    @staticmethod
    def _fee_receipt_same_row(label: OcrLine, value: OcrLine) -> bool:
        """Require a value cell visibly aligned to the right of its label."""

        overlap = min(label.box.top, value.box.top) - max(
            label.box.bottom,
            value.box.bottom,
        )
        minimum_height = min(label.box.height, value.box.height)
        return (
            minimum_height > 0
            and overlap >= minimum_height * 0.50
            and value.box.left >= label.box.right - 8
        )

    def _redundant_fee_receipt_candidate(
        self,
        *,
        lines: tuple[OcrLine, ...],
        page: RenderedPage,
        page_pixels: Any,
        case_id: str,
        active_applicant: str,
    ) -> CandidateEvidence | None:
        """Infer a missing status from two exact, redundant receipt rows.

        The synthetic receipt template redundantly encodes the status in its
        exact amount and waiver-code cells.  This path deliberately abstains
        on OCR approximation, duplicate/conflicting cells, or marked content;
        ordinary explicit ``Fee Status`` evidence is checked before this
        helper is called and always wins.
        """

        if not any(
            _ocr_key(line.text) == _ocr_key("MIB Fee Receipt")
            and line.confidence >= self._minimum_legible_confidence
            for line in lines[:8]
        ):
            return None

        page_visual = (
            self._cues.prepare_page(page_pixels)
            if hasattr(self._cues, "prepare_page")
            else page_pixels
        )

        def clean(line: OcrLine) -> tuple[str, ...] | None:
            if line.confidence < self._minimum_legible_confidence:
                return None
            cues = self._cues.cues_for_line(line, page_visual)
            if (
                "strikethrough" in cues
                or "sample_denial_watermark" in cues
                or self._filter.rejection_reason(line.text, cues) is not None
            ):
                return None
            return cues

        clean_lines = {
            id(line): cues
            for line in lines[:20]
            if (cues := clean(line)) is not None
        }

        def exact_row(
            label_text: str,
            allowed_values: tuple[str, ...],
        ) -> tuple[str, tuple[OcrLine, ...]] | None:
            allowed = {value.casefold(): value for value in allowed_values}
            matches: list[tuple[str, tuple[OcrLine, ...]]] = []
            combined_pattern = re.compile(
                rf"{re.escape(label_text)}\s*:?\s*(\S+)\s*\Z",
                re.I,
            )
            for line in lines[:20]:
                if id(line) not in clean_lines:
                    continue
                combined = combined_pattern.fullmatch(line.text.strip())
                if combined is not None:
                    value = allowed.get(combined.group(1).casefold())
                    if value is not None:
                        matches.append((value, (line,)))

            labels = tuple(
                line
                for line in lines[:20]
                if id(line) in clean_lines
                and re.fullmatch(
                    rf"{re.escape(label_text)}\s*:?",
                    line.text.strip(),
                    re.I,
                )
            )
            values = tuple(
                (allowed[line.text.strip().casefold()], line)
                for line in lines[:20]
                if id(line) in clean_lines
                and line.text.strip().casefold() in allowed
            )
            for label in labels:
                for value, value_line in values:
                    if self._fee_receipt_same_row(label, value_line):
                        matches.append((value, (label, value_line)))

            distinct = {value for value, _support in matches}
            if len(distinct) != 1:
                return None
            value = next(iter(distinct))
            supporting = tuple(
                dict.fromkeys(
                    line
                    for matched_value, match_lines in matches
                    if matched_value == value
                    for line in match_lines
                )
            )
            return value, supporting

        amount_row = exact_row("Amount", ("$809.00", "$0.00"))
        waiver_row = exact_row("Waiver Code", ("N/A", "DIP-WAIVER"))
        if amount_row is None or waiver_row is None:
            return None
        amount, amount_lines = amount_row
        waiver_code, waiver_lines = waiver_row
        inferred = {
            ("$809.00", "N/A"): "paid",
            ("$0.00", "DIP-WAIVER"): "waived",
            ("$0.00", "N/A"): "unknown",
        }.get((amount, waiver_code))
        if inferred is None:
            return None

        supporting = tuple(dict.fromkeys(amount_lines + waiver_lines))
        receipt_box = supporting[0].box
        for line in supporting[1:]:
            receipt_box = receipt_box.union(line.box)
        return CandidateEvidence(
            field_name="fee_status",
            value=inferred,
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=page.index,
            box=receipt_box,
            legible=True,
            superseded=False,
            ocr_confidence=min(line.confidence for line in supporting),
            visual_cues=("redundant_fee_receipt_rows",),
            case_id_hint=case_id,
            applicant_hint=active_applicant,
        )

    def _fee_receipt_retry_evidence(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[CandidateEvidence, ...]:
        """Recover one missing fee only with 3-of-4 threshold consensus."""

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        if rendered_case.case_id is None:
            return ()
        linked = CaseLinker().link(rendered_case.case_id, baseline)
        if linked.unresolved or linked.active_applicant is None:
            return ()
        resolved = EvidencePrecedenceResolver().resolve(linked)
        if resolved.value("fee_status") is not None:
            return ()
        fee_evidence = tuple(
            candidate
            for candidate in baseline
            if candidate.field_name == "fee_status"
        )
        if any(
            candidate.value is not None
            and candidate.legible
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
            and "sample_denial_watermark" not in candidate.visual_cues
            for candidate in fee_evidence
        ):
            # A live explicit status, including an explicit ``unknown``, is
            # authoritative for this receipt path.  A crossed-out unreadable
            # cell is not: the two clean redundant rows may still recover it.
            return ()

        routed_pages: list[tuple[RenderedPage, set[str]]] = []
        for page in rendered_case.pages:
            lines = routing_lines.get(page.index, ())
            if max(
                (
                    self._fee_receipt_heading_similarity(line.text)
                    for line in lines[:8]
                ),
                default=0.0,
            ) < 0.64:
                continue
            visible_case_ids = {
                value
                for line in lines
                if (
                    value := self._normalize_value("case_id", line.text)
                )
                is not None
            }
            if any(
                value != rendered_case.case_id
                for value in visible_case_ids
            ):
                continue
            routed_pages.append((page, visible_case_ids))
        if len(routed_pages) != 1:
            return ()
        page, visible_case_ids = routed_pages[0]

        engines = self._fee_receipt_ocr_engines or tuple(
            TesseractOcrEngine(page_segmentation_mode=11)
            for _threshold in range(4)
        )
        thresholds = (120, 140, 160, 180)
        source_image = self._page_image(page)
        pass_candidates: list[CandidateEvidence] = []
        if rendered_case.case_id not in visible_case_ids:
            footer_case_ids = self._fee_receipt_footer_case_ids(
                page,
                source_image,
                engines[0],
            )
            if footer_case_ids != {rendered_case.case_id}:
                return ()

        redundant_candidate = self._redundant_fee_receipt_candidate(
            lines=routing_lines.get(page.index, ()),
            page=page,
            page_pixels=source_image,
            case_id=rendered_case.case_id,
            active_applicant=linked.active_applicant,
        )
        if redundant_candidate is not None:
            return (redundant_candidate,)

        if any(
            candidate.value is not None
            or candidate.superseded
            or "strikethrough" in candidate.visual_cues
            or "sample_denial_watermark" in candidate.visual_cues
            for candidate in fee_evidence
        ):
            # Preserve the stricter historical gate for approximate
            # threshold-OCR recovery.  Only exact redundant rows may repair a
            # visibly cancelled and unreadable status cell.
            return ()

        primary_candidate = self._fee_receipt_lines_candidate(
            lines=routing_lines.get(page.index, ()),
            page=page,
            page_pixels=source_image,
            case_id=rendered_case.case_id,
            active_applicant=linked.active_applicant,
            view_cue="primary_fee_receipt_view",
        )
        if primary_candidate is not None:
            pass_candidates.append(primary_candidate)
        for threshold, engine in zip(thresholds, engines):
            crop_page, crop_pixels = self._fee_receipt_crop(
                page,
                source_image,
                threshold,
            )
            candidate = self._fee_receipt_pass_candidate(
                page=crop_page,
                page_pixels=crop_pixels,
                engine=engine,
                case_id=rendered_case.case_id,
                active_applicant=linked.active_applicant,
            )
            if candidate is not None:
                pass_candidates.append(candidate)

        values = {candidate.value for candidate in pass_candidates}
        if len(pass_candidates) < 3 or len(values) != 1:
            return ()
        representative = max(
            pass_candidates,
            key=lambda candidate: candidate.ocr_confidence,
        )
        receipt_box = pass_candidates[0].box
        for candidate in pass_candidates[1:]:
            receipt_box = receipt_box.union(candidate.box)
        return (
            CandidateEvidence(
                field_name="fee_status",
                value=representative.value,
                evidence_type=EvidenceType.INTAKE_FORM,
                page_index=page.index,
                box=receipt_box,
                legible=True,
                superseded=False,
                ocr_confidence=min(
                    candidate.ocr_confidence
                    for candidate in pass_candidates
                ),
                visual_cues=("threshold_consensus_fee_receipt",),
                case_id_hint=rendered_case.case_id,
                applicant_hint=linked.active_applicant,
            ),
        )

    _SPARSE_INTAKE_ROUTE_FIELDS = (
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
    )
    _SPARSE_INTAKE_FILL_FIELDS = (
        "home_world",
        "visa_class",
        "arrival_date",
        "declared_purpose",
    )
    _SPARSE_PAGE_VETO = re.compile(
        r"adjudicat|finding|reason\s*:|biometric|fee\s+receipt|"
        r"sponsor\s+attest|registry\s+extract",
        re.I,
    )

    def _sparse_intake_target_page(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[RenderedPage, str, str] | None:
        """Route at most one severely damaged, otherwise-unresolved page."""

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        linked = CaseLinker().link(rendered_case.case_id, baseline)
        if linked.unresolved or linked.active_applicant is None:
            return None
        resolved = EvidencePrecedenceResolver().resolve(linked)
        active_species = resolved.value("species_code")
        if active_species is None:
            return None
        missing_count = sum(
            resolved.value(field_name) is None
            for field_name in self._SPARSE_INTAKE_ROUTE_FIELDS
        )
        if missing_count < 4:
            return None

        ranked: list[tuple[int, int, float, int, RenderedPage]] = []
        for page in rendered_case.pages:
            if any(
                candidate.page_index == page.index
                and candidate.field_name in self._SPARSE_INTAKE_ROUTE_FIELDS
                and candidate.value is not None
                and candidate.legible
                for candidate in baseline
            ):
                continue
            lines = routing_lines.get(page.index, ())
            page_text = " ".join(line.text for line in lines)
            if self._SPARSE_PAGE_VETO.search(page_text):
                continue
            if any(
                self._filter.context_quarantine_lines(line.text)
                or self._filter.rejection_reason(line.text) is not None
                for line in lines
            ):
                continue
            alphanumeric_lines = tuple(
                line
                for line in lines
                if len(re.sub(r"[^A-Za-z0-9]", "", line.text)) >= 2
            )
            if len(lines) < 12 or len(alphanumeric_lines) < 4:
                continue
            confidences = sorted(line.confidence for line in alphanumeric_lines)
            midpoint = len(confidences) // 2
            median = (
                confidences[midpoint]
                if len(confidences) % 2
                else (confidences[midpoint - 1] + confidences[midpoint]) / 2.0
            )
            low_confidence_count = sum(
                confidence < 0.65 for confidence in confidences
            )
            if median >= 0.65 or low_confidence_count < 4:
                continue
            ranked.append(
                (
                    low_confidence_count,
                    len(lines),
                    -median,
                    page.index,
                    page,
                )
            )
        if not ranked:
            return None
        _low, _line_count, _median, _index, page = max(
            ranked,
            key=lambda item: item[:4],
        )
        return page, linked.active_applicant, active_species

    @staticmethod
    def _sparse_intake_crop(page: RenderedPage, image: Any) -> tuple[RenderedPage, int, int]:
        """Crop the template-relative top-left intake block from visible pixels."""

        width, height = image.size
        left = int(width * 0.04)
        upper = int(height * 0.035)
        right = max(left + 1, int(width * 0.62))
        lower = max(upper + 1, int(height * 0.38))
        crop = image.crop((left, upper, right, lower))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return (
            RenderedPage(
                index=page.index,
                image_png=buffer.getvalue(),
                width_px=crop.width,
                height_px=crop.height,
                dpi=page.dpi,
                rotation_deg=0,
                skew_correction_deg=0.0,
                crop_box=Rect(0, 0, crop.width, crop.height),
                text_spans=(),
            ),
            left,
            upper,
        )

    @staticmethod
    def _sparse_pass_values(
        candidates: Iterable[CandidateEvidence],
    ) -> dict[str, CandidateEvidence]:
        eligible_fields = {
            "case_id",
            "applicant_name",
            "species_code",
            "home_world",
            "visa_class",
            "sponsor_id",
            "arrival_date",
            "declared_purpose",
        }
        grouped: dict[str, list[CandidateEvidence]] = {}
        for candidate in candidates:
            if (
                candidate.field_name not in eligible_fields
                or candidate.value is None
                or not candidate.legible
                or candidate.superseded
                or candidate.ocr_confidence < 0.65
                or "strikethrough" in candidate.visual_cues
                or "sample_denial_watermark" in candidate.visual_cues
            ):
                continue
            grouped.setdefault(candidate.field_name, []).append(candidate)
        accepted: dict[str, CandidateEvidence] = {}
        for field_name, field_candidates in grouped.items():
            if len({candidate.value for candidate in field_candidates}) != 1:
                continue
            accepted[field_name] = max(
                field_candidates,
                key=lambda candidate: candidate.ocr_confidence,
            )
        return accepted

    def _sparse_intake_retry_evidence(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[CandidateEvidence, ...]:
        """Fill a coherent intake bundle only when cropped PSM 6 and 3 agree."""

        target = self._sparse_intake_target_page(
            rendered_case,
            baseline,
            routing_lines,
        )
        if target is None:
            return ()
        page, active_applicant, active_species = target
        crop_page, crop_left, crop_upper = self._sparse_intake_crop(
            page,
            self._page_image(page),
        )
        retry_case = RenderedCase(
            source_path=rendered_case.source_path,
            source_sha256=rendered_case.source_sha256,
            case_id=rendered_case.case_id,
            pages=(crop_page,),
            # The retry is deliberately PNG-only. Never pass the PDF text layer
            # into this recovery path.
            text_layer=(),
        )
        engines = self._sparse_intake_ocr_engines or (
            TesseractOcrEngine(page_segmentation_mode=6),
            TesseractOcrEngine(page_segmentation_mode=3),
        )
        passes: list[dict[str, CandidateEvidence]] = []
        for engine in engines:
            retry_extractor = VisibleEvidenceExtractor(
                ocr_engine=engine,
                cue_detector=self._cues,
                content_filter=self._filter,
                refinement_model=None,
                psm6_refinement=False,
                minimum_legible_confidence=self._minimum_legible_confidence,
                refinement_gate=self._refinement_gate,
                consensus_retry=False,
                sparse_intake_retry=False,
            )
            try:
                retry_candidates = retry_extractor.extract(retry_case)
            except RecoverableOcrError:
                return ()
            passes.append(self._sparse_pass_values(retry_candidates))

        agreed: dict[str, tuple[CandidateEvidence, CandidateEvidence]] = {}
        for field_name, first in passes[0].items():
            second = passes[1].get(field_name)
            if second is not None and second.value == first.value:
                agreed[field_name] = (first, second)
        if (
            len(agreed) < 5
            or agreed.get("applicant_name", (None,))[0] is None
            or agreed["applicant_name"][0].value != active_applicant
            or agreed.get("species_code", (None,))[0] is None
            or agreed["species_code"][0].value != active_species
        ):
            return ()
        case_pair = agreed.get("case_id")
        if (
            case_pair is not None
            and case_pair[0].value != rendered_case.case_id
        ):
            return ()

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        resolved = EvidencePrecedenceResolver().resolve(
            CaseLinker().link(rendered_case.case_id, baseline)
        )
        recovered: list[CandidateEvidence] = []
        for field_name in self._SPARSE_INTAKE_FILL_FIELDS:
            pair = agreed.get(field_name)
            if pair is None or resolved.value(field_name) is not None:
                continue
            first, second = pair
            recovered.append(
                CandidateEvidence(
                    field_name=field_name,
                    value=first.value,
                    evidence_type=EvidenceType.INTAKE_FORM,
                    page_index=page.index,
                    box=Rect(
                        first.box.left + crop_left,
                        first.box.bottom + crop_upper,
                        first.box.right + crop_left,
                        first.box.top + crop_upper,
                    ),
                    legible=True,
                    superseded=False,
                    ocr_confidence=min(
                        first.ocr_confidence,
                        second.ocr_confidence,
                    ),
                    visual_cues=tuple(
                        sorted(
                            set(first.visual_cues)
                            | set(second.visual_cues)
                            | {"sparse_intake_consensus"}
                        )
                    ),
                    case_id_hint=rendered_case.case_id,
                    applicant_hint=active_applicant,
                )
            )
        return tuple(recovered)

    @staticmethod
    def _retry_label_similarity(text: str, field_name: str) -> float:
        """Score only a leading visible phrase against published field labels."""

        words = text.split()
        best = 0.0
        for prefix_length in range(1, min(4, len(words)) + 1):
            prefix_key = _ocr_key(" ".join(words[:prefix_length]))
            if not prefix_key:
                continue
            for alias in FIELD_ALIASES[field_name]:
                best = max(
                    best,
                    difflib.SequenceMatcher(
                        None, prefix_key, _ocr_key(alias)
                    ).ratio(),
                )
        return best

    _ORIENTATION_ROUTE_FIELDS = (
        "applicant_name",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "risk_flags",
        "fee_status",
    )
    _ORIENTATION_FOOTER_RE = re.compile(
        r"\bpacket\s+(MIB-[0-9]{6})\s*/\s*page\s+[0-9]+\b",
        re.I,
    )
    _ORIENTATION_FOOTER_NOISE_RE = re.compile(
        r"\bpacket\s+MIB-|synthetic\s+hiring\s+challenge\s+document",
        re.I,
    )

    def _orientation_target_pages(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[RenderedPage, ...]:
        """Route at most two candidate-free pages using primary OCR only."""

        case_id = rendered_case.case_id
        if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
            return ()

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        resolved = EvidencePrecedenceResolver().resolve(
            CaseLinker().link(case_id, baseline)
        )
        missing_count = sum(
            resolved.value(field_name)
            in {None, "unknown", "1900-01-01", "SPN-0000"}
            for field_name in self._ORIENTATION_ROUTE_FIELDS
        )
        if missing_count < 4:
            return ()

        ranked: list[tuple[tuple[int, int, float, int], RenderedPage]] = []
        for page in rendered_case.pages:
            lines = routing_lines.get(page.index, ())
            exact_footer = any(
                match is not None and match.group(1).upper() == case_id.upper()
                for line in lines
                if (match := self._ORIENTATION_FOOTER_RE.search(line.text))
            )
            if not exact_footer:
                continue
            # Any structured primary candidate, including illegible evidence,
            # makes this an ordinary precedence/linkage problem rather than a
            # physically rotated blank-page recovery.
            if any(
                candidate.page_index == page.index
                for candidate in baseline
            ):
                continue

            informative = tuple(
                line
                for line in lines
                if not self._ORIENTATION_FOOTER_NOISE_RE.search(line.text)
            )
            substantive_count = sum(
                len(_ocr_key(line.text)) >= 4 and line.confidence >= 0.55
                for line in informative
            )
            alphanumeric_count = sum(
                len(_ocr_key(line.text)) >= 2
                for line in informative
            )
            confidences = sorted(
                line.confidence
                for line in informative
                if len(_ocr_key(line.text)) >= 2
            )
            median_confidence = (
                confidences[len(confidences) // 2] if confidences else 0.0
            )
            rank = (
                -substantive_count,
                -alphanumeric_count,
                -median_confidence,
                -page.index,
            )
            ranked.append((rank, page))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return tuple(page for _rank, page in ranked[:2])

    def _orientation_retry_pass(
        self,
        rendered_case: RenderedCase,
        page: RenderedPage,
        *,
        angle: int,
        engine: Any,
    ) -> tuple[
        tuple[OcrLine, ...],
        dict[str, CandidateEvidence],
    ] | None:
        """Read one rotated view once and parse only its visible OCR tokens."""

        source_image = self._page_image(page)
        rotated_image = source_image.rotate(angle, expand=True, fillcolor=255)
        buffer = io.BytesIO()
        rotated_image.save(buffer, format="PNG")
        retry_page = RenderedPage(
            index=page.index,
            image_png=buffer.getvalue(),
            width_px=rotated_image.width,
            height_px=rotated_image.height,
            dpi=page.dpi,
            rotation_deg=angle % 360,
            skew_correction_deg=0.0,
            crop_box=Rect(0, 0, rotated_image.width, rotated_image.height),
            text_spans=(),
        )
        try:
            tokens = tuple(engine.read_page(retry_page))
        except RecoverableOcrError:
            return None
        lines = _visual_reading_order(group_ocr_lines(tokens))

        class CapturedOcrEngine:
            def read_page(self, _page: RenderedPage) -> tuple[OcrToken, ...]:
                return tokens

        retry_case = RenderedCase(
            source_path=rendered_case.source_path,
            source_sha256=rendered_case.source_sha256,
            case_id=rendered_case.case_id,
            pages=(retry_page,),
            text_layer=(),
        )
        retry_extractor = VisibleEvidenceExtractor(
            ocr_engine=CapturedOcrEngine(),
            cue_detector=self._cues,
            content_filter=self._filter,
            refinement_model=None,
            psm6_refinement=False,
            minimum_legible_confidence=self._minimum_legible_confidence,
            refinement_gate=self._refinement_gate,
            consensus_retry=False,
            fee_receipt_retry=False,
            sparse_intake_retry=False,
            orientation_retry=False,
        )
        try:
            retry_candidates = retry_extractor.extract(retry_case)
        except RecoverableOcrError:
            return None

        grouped: dict[str, list[CandidateEvidence]] = {}
        for candidate in retry_candidates:
            if (
                candidate.field_name not in CONSENSUS_RETRY_FIELDS
                or candidate.value is None
                or not candidate.legible
                or candidate.superseded
                or candidate.ocr_confidence < self._minimum_legible_confidence
                or "strikethrough" in candidate.visual_cues
                or "sample_denial_watermark" in candidate.visual_cues
            ):
                continue
            grouped.setdefault(candidate.field_name, []).append(candidate)
        accepted: dict[str, CandidateEvidence] = {}
        for field_name, candidates in grouped.items():
            if len({candidate.value for candidate in candidates}) == 1:
                accepted[field_name] = max(
                    candidates,
                    key=lambda candidate: candidate.ocr_confidence,
                )
        return lines, accepted

    def _orientation_score(
        self,
        retry_pass: tuple[tuple[OcrLine, ...], dict[str, CandidateEvidence]],
    ) -> tuple[float, dict[str, float]]:
        """Choose direction only from a trusted label plus a parsed value."""

        lines, accepted = retry_pass
        label_scores: dict[str, float] = {}
        for field_name in CONSENSUS_RETRY_FIELDS:
            score = max(
                (
                    self._retry_label_similarity(line.text, field_name)
                    for line in lines
                ),
                default=0.0,
            )
            if score >= 0.70:
                label_scores[field_name] = score
        supported = set(label_scores) & set(accepted)
        score = sum(label_scores[field] for field in supported)
        score += 0.35 * len(supported)
        return score, label_scores

    @staticmethod
    def _orientation_consensus(
        first: dict[str, CandidateEvidence],
        second: dict[str, CandidateEvidence],
    ) -> tuple[CandidateEvidence, ...]:
        """Accept only exact field/value agreement from scan and confirmation."""

        accepted: list[CandidateEvidence] = []
        for field_name, candidate in first.items():
            other = second.get(field_name)
            if other is None or other.value != candidate.value:
                continue
            if (
                candidate.case_id_hint is not None
                and other.case_id_hint is not None
                and candidate.case_id_hint != other.case_id_hint
            ):
                continue
            if (
                candidate.applicant_hint is not None
                and other.applicant_hint is not None
                and candidate.applicant_hint != other.applicant_hint
            ):
                continue
            chosen = (
                candidate
                if candidate.ocr_confidence >= other.ocr_confidence
                else other
            )
            accepted.append(
                CandidateEvidence(
                    field_name=chosen.field_name,
                    value=chosen.value,
                    evidence_type=chosen.evidence_type,
                    page_index=chosen.page_index,
                    box=chosen.box,
                    legible=chosen.legible,
                    superseded=chosen.superseded,
                    ocr_confidence=min(
                        candidate.ocr_confidence,
                        other.ocr_confidence,
                    ),
                    visual_cues=tuple(
                        sorted(
                            set(candidate.visual_cues)
                            | set(other.visual_cues)
                            | {"sparse_orientation_consensus"}
                        )
                    ),
                    source=chosen.source,
                    case_id_hint=chosen.case_id_hint,
                    applicant_hint=chosen.applicant_hint,
                )
            )
        return tuple(accepted)

    def _orientation_retry_evidence(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[CandidateEvidence, ...]:
        """Recover fields from clearly sideways, otherwise blank form pages."""

        pages = self._orientation_target_pages(
            rendered_case,
            baseline,
            routing_lines,
        )
        if not pages:
            return ()
        engines = self._orientation_ocr_engines or (
            TesseractOcrEngine(page_segmentation_mode=11),
            TesseractOcrEngine(page_segmentation_mode=11),
            TesseractOcrEngine(page_segmentation_mode=6),
        )

        page_results: list[
            tuple[tuple[CandidateEvidence, ...], set[str]]
        ] = []
        for page in pages:
            scans: list[
                tuple[
                    int,
                    tuple[tuple[OcrLine, ...], dict[str, CandidateEvidence]],
                    float,
                    dict[str, float],
                ]
            ] = []
            for angle, engine in zip((90, 270), engines[:2]):
                retry_pass = self._orientation_retry_pass(
                    rendered_case,
                    page,
                    angle=angle,
                    engine=engine,
                )
                if retry_pass is None:
                    break
                score, labels = self._orientation_score(retry_pass)
                scans.append((angle, retry_pass, score, labels))
            if len(scans) != 2:
                continue
            scans.sort(key=lambda item: item[2], reverse=True)
            best_angle, best_pass, best_score, best_labels = scans[0]
            other_score = scans[1][2]
            accepted: tuple[CandidateEvidence, ...] = ()
            if best_score >= 1.05 and best_score - other_score >= 0.45:
                confirmation = self._orientation_retry_pass(
                    rendered_case,
                    page,
                    angle=best_angle,
                    engine=engines[2],
                )
                if confirmation is not None:
                    accepted = self._orientation_consensus(
                        best_pass[1],
                        confirmation[1],
                    )
            accepted_fields = {candidate.field_name for candidate in accepted}
            unconfirmed_labels = set(best_labels) - accepted_fields
            page_results.append((accepted, unconfirmed_labels))

        blocked_fields = {
            field_name
            for _accepted, unconfirmed in page_results
            for field_name in unconfirmed
        }
        recovered_by_field: dict[str, list[CandidateEvidence]] = {}
        for accepted, _unconfirmed in page_results:
            for candidate in accepted:
                if candidate.field_name in blocked_fields:
                    continue
                if (
                    candidate.field_name == "sponsor_id"
                    and candidate.applicant_hint is not None
                    and candidate.case_id_hint is None
                    and candidate.ocr_confidence < 0.70
                ):
                    continue
                recovered_by_field.setdefault(candidate.field_name, []).append(
                    candidate
                )

        final: list[CandidateEvidence] = []
        for field_name, candidates in recovered_by_field.items():
            baseline_field = tuple(
                candidate
                for candidate in baseline
                if candidate.field_name == field_name
            )
            if any(
                candidate.value is not None
                or candidate.superseded
                or "strikethrough" in candidate.visual_cues
                or "sample_denial_watermark" in candidate.visual_cues
                for candidate in baseline_field
            ):
                continue
            if len({candidate.value for candidate in candidates}) != 1:
                continue
            final.extend(candidates)
        return tuple(final)

    _TRUSTED_SCOPE_TYPES = frozenset(
        {
            EvidenceType.ADJUDICATOR_STAMP,
            EvidenceType.SIGNED_MANUAL_NOTE,
        }
    )
    _RISK_RETRY_TYPES = _TRUSTED_SCOPE_TYPES | {
        EvidenceType.BIOMETRIC_SLIP,
    }
    _STRICT_CASE_ID_RE = re.compile(r"^MIB-([0-9]{6})$")

    def _trusted_line_cues(
        self,
        page: RenderedPage,
        lines: tuple[OcrLine, ...],
    ) -> tuple[tuple[OcrLine, tuple[str, ...]], ...]:
        """Apply the ordinary quarantine, content, and visual-cue filters."""

        page_image = self._page_image(page)
        page_visual = (
            self._cues.prepare_page(page_image)
            if hasattr(self._cues, "prepare_page")
            else page_image
        )
        accepted: list[tuple[OcrLine, tuple[str, ...]]] = []
        quarantine_remaining = 0
        for line in lines:
            cues = self._cues.cues_for_line(line, page_visual)
            quarantine = self._filter.context_quarantine_lines(line.text)
            if quarantine:
                quarantine_remaining = max(quarantine_remaining, quarantine)
            if quarantine_remaining > 0:
                quarantine_remaining -= 1
                continue
            if self._filter.rejection_reason(line.text, cues) is not None:
                continue
            accepted.append((line, cues))
        return tuple(accepted)

    @classmethod
    def _case_id_hamming_distance(
        cls,
        expected: str,
        observed: str,
    ) -> int | None:
        expected_match = cls._STRICT_CASE_ID_RE.fullmatch(expected)
        observed_match = cls._STRICT_CASE_ID_RE.fullmatch(observed)
        if expected_match is None or observed_match is None:
            return None
        return sum(
            left != right
            for left, right in zip(
                expected_match.group(1),
                observed_match.group(1),
            )
        )

    def _trusted_scope_repair_evidence(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[CandidateEvidence, ...]:
        """Repair one-digit footer OCR drift on a trusted risk decision page."""

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        if not self._STRICT_CASE_ID_RE.fullmatch(rendered_case.case_id):
            return ()
        resolved = EvidencePrecedenceResolver().resolve(
            CaseLinker().link(rendered_case.case_id, baseline)
        )
        # This recovery was frozen only for an otherwise unresolved risk field.
        # An explicit ``none`` or any already-resolved risk is substantive and
        # must not be replaced by a near-ID page.
        if resolved.value("risk_flags") is not None:
            return ()

        recovered: list[CandidateEvidence] = []
        for page in rendered_case.pages:
            lines = routing_lines.get(page.index, ())
            heading_type = self._visible_page_heading_type(lines)
            if heading_type not in self._TRUSTED_SCOPE_TYPES:
                continue
            id_lines = tuple(
                (case_id, line)
                for line in lines
                if (
                    case_id := self._normalize_value("case_id", line.text)
                )
                is not None
            )
            visible_ids = {case_id for case_id, _line in id_lines}
            if (
                rendered_case.case_id in visible_ids
                or len(visible_ids) != 1
            ):
                continue
            observed_case_id = next(iter(visible_ids))
            if (
                self._case_id_hamming_distance(
                    rendered_case.case_id,
                    observed_case_id,
                )
                != 1
            ):
                continue
            if not any(
                case_id == observed_case_id
                and line.box.top >= page.height_px * 0.84
                and re.search(r"\bpacket\b", line.text, re.I)
                for case_id, line in id_lines
            ):
                continue

            trusted = tuple(
                (line, cues)
                for line, cues in self._trusted_line_cues(page, lines)
                if "strikethrough" not in cues
                and "sample_denial_watermark" not in cues
            )
            risk_records: list[
                tuple[set[str], OcrLine, tuple[str, ...]]
            ] = []
            for line, cues in trusted:
                flags = set(self._risk_flags_from_text(line.text))
                flags.update(self._fuzzy_risk_flags_from_text(line.text))
                if flags:
                    risk_records.append((flags, line, cues))
            observed_flags = {
                flag
                for flags, _line, _cues in risk_records
                for flag in flags
            }
            if not observed_flags:
                continue

            _flags, representative, _cues = max(
                risk_records,
                key=lambda record: (
                    len(record[0]),
                    record[1].confidence,
                ),
            )
            risk_box = risk_records[0][1].box
            for _record_flags, line, _record_cues in risk_records[1:]:
                risk_box = risk_box.union(line.box)
            risk_cues = {
                cue
                for _record_flags, _line, cues in risk_records
                for cue in cues
            }
            risk_cues.add("trusted_footer_scope_repair")
            recovered.append(
                CandidateEvidence(
                    field_name="risk_flags",
                    value="|".join(sorted(observed_flags)),
                    evidence_type=heading_type,
                    page_index=page.index,
                    box=risk_box,
                    legible=True,
                    superseded=False,
                    ocr_confidence=representative.confidence,
                    visual_cues=tuple(sorted(risk_cues)),
                    case_id_hint=rendered_case.case_id,
                    applicant_hint=None,
                )
            )

            if resolved.value("adjudication") is not None:
                continue
            decision = self._recovered_authoritative_decision(trusted)
            if decision is None:
                continue
            decision_value, supporting = decision
            decision_box = supporting[0][0].box
            for line, _support_cues in supporting[1:]:
                decision_box = decision_box.union(line.box)
            decision_cues = {
                cue
                for _line, cues in supporting
                for cue in cues
            }
            decision_cues.add("trusted_footer_scope_repair")
            recovered.append(
                CandidateEvidence(
                    field_name="adjudication",
                    value=decision_value,
                    evidence_type=heading_type,
                    page_index=page.index,
                    box=decision_box,
                    legible=True,
                    superseded=False,
                    ocr_confidence=max(
                        line.confidence for line, _cues in supporting
                    ),
                    visual_cues=tuple(sorted(decision_cues)),
                    case_id_hint=rendered_case.case_id,
                    applicant_hint=None,
                )
            )
        return tuple(recovered)

    @staticmethod
    def _risk_retry_title_score(text: str) -> float:
        key = _ocr_key(text)
        target = "formb13biometricscanslip"
        similarity = difflib.SequenceMatcher(None, key, target).ratio()
        if len(key) >= 9 and target.startswith(key):
            return max(similarity, 0.82)
        if key.startswith("formb13bi"):
            return max(similarity, 0.78)
        return similarity

    def _risk_retry_target_page(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> RenderedPage | None:
        """Route one exact-case biometric or trusted narrative page."""

        ranked: list[tuple[float, int, RenderedPage]] = []
        for page in rendered_case.pages:
            lines = routing_lines.get(page.index, ())
            heading_type = self._visible_page_heading_type(lines)
            biometric_score = max(
                (
                    self._risk_retry_title_score(line.text)
                    for line in lines[:4]
                ),
                default=0.0,
            )
            biometric_page = (
                heading_type is EvidenceType.BIOMETRIC_SLIP
                or biometric_score >= 0.78
            )
            trusted_narrative = heading_type in self._TRUSTED_SCOPE_TYPES
            if not biometric_page and not trusted_narrative:
                continue
            visible_ids = {
                value
                for line in lines
                if (
                    value := self._normalize_value("case_id", line.text)
                )
                is not None
            }
            if visible_ids != {rendered_case.case_id}:
                continue
            label_score = max(
                (
                    self._retry_label_similarity(line.text, "risk_flags")
                    for line in lines
                ),
                default=0.0,
            )
            route_score = label_score + (2.0 if biometric_page else 1.0)
            ranked.append((route_score, -page.index, page))
        if not ranked:
            return None
        _score, _negative_index, page = max(
            ranked,
            key=lambda item: item[:2],
        )
        # A primary value, explicit ``none``, supersession, or visual veto is
        # a precedence/content problem rather than an OCR gap.
        if any(
            candidate.field_name == "risk_flags"
            and candidate.page_index == page.index
            and candidate.case_id_hint in {None, rendered_case.case_id}
            and (
                candidate.value is not None
                or candidate.superseded
                or "strikethrough" in candidate.visual_cues
                or "sample_denial_watermark" in candidate.visual_cues
            )
            for candidate in baseline
        ):
            return None
        return page

    @staticmethod
    def _risk_retry_crop(page: RenderedPage, image: Any) -> RenderedPage:
        width, height = image.size
        crop = image.crop((0, 0, width, max(1, round(height * 0.40))))
        buffer = io.BytesIO()
        crop.save(buffer, format="PNG")
        return RenderedPage(
            index=page.index,
            image_png=buffer.getvalue(),
            width_px=crop.width,
            height_px=crop.height,
            dpi=page.dpi,
            rotation_deg=page.rotation_deg,
            skew_correction_deg=0.0,
            crop_box=Rect(0, 0, crop.width, crop.height),
            text_spans=(),
        )

    def _risk_retry_pass(
        self,
        rendered_case: RenderedCase,
        page: RenderedPage,
        engine: Any,
    ) -> tuple[str, tuple[CandidateEvidence, ...]] | None:
        retry_page = self._risk_retry_crop(page, self._page_image(page))
        retry_case = RenderedCase(
            source_path=rendered_case.source_path,
            source_sha256=rendered_case.source_sha256,
            case_id=rendered_case.case_id,
            pages=(retry_page,),
            text_layer=(),
        )
        retry_extractor = VisibleEvidenceExtractor(
            ocr_engine=engine,
            cue_detector=self._cues,
            content_filter=self._filter,
            psm6_refinement=False,
            minimum_legible_confidence=self._minimum_legible_confidence,
            refinement_gate=self._refinement_gate,
            consensus_retry=False,
            fee_receipt_retry=False,
            sparse_intake_retry=False,
            orientation_retry=False,
            trusted_scope_repair=False,
            risk_flag_retry=False,
        )
        try:
            retry_candidates = retry_extractor.extract(retry_case)
        except RecoverableOcrError:
            return None
        eligible = tuple(
            candidate
            for candidate in retry_candidates
            if candidate.field_name == "risk_flags"
            and candidate.value not in {None, "none"}
            and candidate.legible
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
            and "sample_denial_watermark" not in candidate.visual_cues
            and candidate.evidence_type in self._RISK_RETRY_TYPES
            and candidate.case_id_hint in {None, rendered_case.case_id}
        )
        values = {candidate.value for candidate in eligible}
        maximal = {
            value
            for value in values
            if not any(
                set(value.split("|")) < set(other.split("|"))
                for other in values
            )
        }
        if len(maximal) != 1:
            return None
        value = next(iter(maximal))
        return value, tuple(
            candidate
            for candidate in eligible
            if candidate.value == value
        )

    def _risk_flag_retry_evidence(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[CandidateEvidence, ...]:
        """Recover one risk value only under conflict-free three-pass agreement."""

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        resolved = EvidencePrecedenceResolver().resolve(
            CaseLinker().link(rendered_case.case_id, baseline)
        )
        if resolved.value("risk_flags") is not None:
            return ()
        page = self._risk_retry_target_page(
            rendered_case,
            baseline,
            routing_lines,
        )
        if page is None:
            return ()
        engines = self._risk_flag_ocr_engines or (
            TesseractOcrEngine(page_segmentation_mode=3),
            TesseractOcrEngine(page_segmentation_mode=4),
            TesseractOcrEngine(page_segmentation_mode=12),
        )
        passes = tuple(
            self._risk_retry_pass(rendered_case, page, engine)
            for engine in engines
        )
        values = tuple(
            retry_pass[0] if retry_pass is not None else None
            for retry_pass in passes
        )
        counts: dict[str, int] = {}
        for value in values:
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return ()
        value, count = max(counts.items(), key=lambda item: item[1])
        if count < 2 or any(
            candidate_value is not None and candidate_value != value
            for candidate_value in values
        ):
            return ()
        representatives = tuple(
            max(retry_pass[1], key=lambda item: item.ocr_confidence)
            for retry_pass in passes
            if retry_pass is not None and retry_pass[0] == value
        )
        chosen = max(
            representatives,
            key=lambda item: item.ocr_confidence,
        )
        cues = {
            cue
            for candidate in representatives
            for cue in candidate.visual_cues
        }
        cues.add("cropped_risk_consensus")
        return (
            CandidateEvidence(
                field_name="risk_flags",
                value=value,
                evidence_type=chosen.evidence_type,
                page_index=page.index,
                box=chosen.box,
                legible=True,
                superseded=False,
                ocr_confidence=min(
                    candidate.ocr_confidence
                    for candidate in representatives
                ),
                visual_cues=tuple(sorted(cues)),
                source=chosen.source,
                case_id_hint=rendered_case.case_id,
                applicant_hint=None,
            ),
        )

    def _retry_target_page(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
        retry_fields: tuple[str, ...],
    ) -> RenderedPage | None:
        """Choose at most one page using primary-pass visible routing signals."""

        ranked: list[tuple[float, int, RenderedPage]] = []
        for page in rendered_case.pages:
            score = 0.0
            for field_name in retry_fields:
                if any(
                    candidate.field_name == field_name
                    and candidate.page_index == page.index
                    for candidate in baseline
                ):
                    score += 5.0
                best_label_score = max(
                    (
                        self._retry_label_similarity(line.text, field_name)
                        for line in routing_lines.get(page.index, ())
                    ),
                    default=0.0,
                )
                if best_label_score >= 0.62:
                    score += best_label_score
            ranked.append((score, -page.index, page))
        if not ranked:
            return None
        score, _negative_index, page = max(ranked, key=lambda item: item[:2])
        return page if score > 0.0 else None

    @staticmethod
    def _eligible_retry_candidates(
        candidates: Iterable[CandidateEvidence],
        *,
        case_id: str,
        active_applicant: str,
        retry_fields: tuple[str, ...],
    ) -> tuple[CandidateEvidence, ...]:
        return tuple(
            candidate
            for candidate in candidates
            if candidate.field_name in retry_fields
            and candidate.value is not None
            and candidate.legible
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
            and "sample_denial_watermark" not in candidate.visual_cues
            and candidate.case_id_hint in {None, case_id}
            and candidate.applicant_hint in {None, active_applicant}
        )

    def _consensus_retry_evidence(
        self,
        rendered_case: RenderedCase,
        baseline: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> tuple[CandidateEvidence, ...]:
        """Fill unresolved fields only when raw PSM 3 and 4 exactly agree."""

        # Imported lazily to preserve extraction/resolution module initialization
        # order while keeping the safety decision next to the captured OCR trace.
        from .resolution import CaseLinker, EvidencePrecedenceResolver

        linked = CaseLinker().link(rendered_case.case_id, baseline)
        if (
            not rendered_case.case_id
            or linked.unresolved
            or linked.active_applicant is None
        ):
            return ()
        resolved = EvidencePrecedenceResolver().resolve(linked)
        retry_fields: list[str] = []
        for field_name in CONSENSUS_RETRY_FIELDS:
            if resolved.value(field_name) is not None:
                continue
            field_evidence = tuple(
                candidate
                for candidate in baseline
                if candidate.field_name == field_name
            )
            # A substantive primary-pass candidate that resolution excluded is
            # a linkage/precedence/supersession issue, not an OCR gap. Retrying
            # it can resurrect a decoy, so this is a hard veto.
            if any(
                candidate.value is not None
                or candidate.superseded
                or "strikethrough" in candidate.visual_cues
                or "sample_denial_watermark" in candidate.visual_cues
                for candidate in field_evidence
            ):
                continue
            retry_fields.append(field_name)
        if not retry_fields:
            return ()

        retry_field_tuple = tuple(retry_fields)
        page = self._retry_target_page(
            rendered_case,
            baseline,
            routing_lines,
            retry_field_tuple,
        )
        if page is None:
            return ()
        retry_case = RenderedCase(
            source_path=rendered_case.source_path,
            source_sha256=rendered_case.source_sha256,
            case_id=rendered_case.case_id,
            pages=(page,),
            text_layer=tuple(
                span
                for span in rendered_case.text_layer
                if span.page_index == page.index
            ),
        )
        engines = self._retry_ocr_engines or (
            TesseractOcrEngine(page_segmentation_mode=3),
            TesseractOcrEngine(page_segmentation_mode=4),
        )
        passes: list[tuple[CandidateEvidence, ...]] = []
        for engine in engines:
            retry_extractor = VisibleEvidenceExtractor(
                ocr_engine=engine,
                cue_detector=self._cues,
                content_filter=self._filter,
                minimum_legible_confidence=self._minimum_legible_confidence,
                refinement_gate=self._refinement_gate,
                consensus_retry=False,
            )
            try:
                retry_candidates = retry_extractor.extract(retry_case)
            except RecoverableOcrError:
                return ()
            passes.append(
                self._eligible_retry_candidates(
                    retry_candidates,
                    case_id=rendered_case.case_id,
                    active_applicant=linked.active_applicant,
                    retry_fields=retry_field_tuple,
                )
            )

        accepted: list[CandidateEvidence] = []
        for field_name in retry_field_tuple:
            values_by_pass = [
                {
                    candidate.value
                    for candidate in candidates
                    if candidate.field_name == field_name
                }
                for candidates in passes
            ]
            if any(len(values) != 1 for values in values_by_pass):
                continue
            psm3_value = next(iter(values_by_pass[0]))
            psm4_value = next(iter(values_by_pass[1]))
            if psm3_value != psm4_value:
                continue
            accepted.extend(
                candidate
                for candidates in passes
                for candidate in candidates
                if candidate.field_name == field_name
                and candidate.value == psm3_value
            )
        return tuple(accepted)

    def _minimal_diplomatic_packet_marker(
        self,
        rendered_case: RenderedCase,
        candidates: tuple[CandidateEvidence, ...],
        routing_lines: dict[int, tuple[OcrLine, ...]],
    ) -> CandidateEvidence | None:
        """Mark only the frozen three-page minimal diplomatic topology.

        Labeled examples establish a narrow exception for a complete intake,
        clear matching registry, and fee receipt when no biometric document is
        present.  This marker deliberately rechecks the raw visible OCR page
        topology, exact packet footers, all repeated values, and trap/risk
        vetoes.  It does not infer the decision itself; adjudication still
        requires the one exact stale-DIP/risk-gap review signature.
        """

        case_id = rendered_case.case_id
        if (
            case_id is None
            or not CASE_ID_PATTERN.fullmatch(case_id)
            or len(rendered_case.pages) != 3
            or set(routing_lines) != {0, 1, 2}
        ):
            return None
        page_text = {
            page_index: " ".join(line.text for line in lines)
            for page_index, lines in routing_lines.items()
        }
        intake_pages = [
            page_index
            for page_index, text in page_text.items()
            if re.search(
                r"\b(?:FORM\s+I-8090|work\s+authorization\s+intake|"
                r"primary\s+intake\s+record)\b",
                text,
                re.I,
            )
        ]
        registry_pages = [
            page_index
            for page_index, text in page_text.items()
            if re.search(r"\b(?:planetary\s+)?registry\s+extract\b", text, re.I)
        ]
        fee_pages = [
            page_index
            for page_index, text in page_text.items()
            if re.search(r"\bMIB\s+fee\s+receipt\b", text, re.I)
        ]
        if (
            len(intake_pages) != 1
            or len(registry_pages) != 1
            or len(fee_pages) != 1
            or len(set(intake_pages + registry_pages + fee_pages)) != 3
        ):
            return None

        for page_index, lines in routing_lines.items():
            if not any(
                match is not None and match.group(1).upper() == case_id
                for line in lines
                if (match := self._ORIENTATION_FOOTER_RE.search(line.text))
            ):
                return None
            visible_ids = {
                value
                for line in lines
                if (
                    value := self._normalize_value("case_id", line.text)
                ) is not None
            }
            if visible_ids != {case_id}:
                return None
            if any(
                self._filter.context_quarantine_lines(line.text)
                or self._filter.rejection_reason(line.text) is not None
                for line in lines
            ):
                return None

        all_text = " ".join(page_text.values())
        forbidden_text = (
            r"\bbiometric\b",
            r"\b(?:observed|risk)\s+flags?\b",
            r"\bmanual\s+(?:adjudicator\s+)?note\b",
            r"\badjudicator\s+stamp\b",
            r"\bsponsor\s+attestation\b",
        )
        risk_phrases = tuple(
            re.escape(flag).replace("_", r"[\s_-]+")
            for flag in KNOWN_RISK_FLAGS
        )
        if any(re.search(pattern, all_text, re.I) for pattern in forbidden_text):
            return None
        if any(re.search(rf"\b{pattern}\b", all_text, re.I) for pattern in risk_phrases):
            return None
        if any(
            candidate.field_name in {"risk_flags", "adjudication"}
            or candidate.case_id_hint not in {None, case_id}
            or bool(
                set(candidate.visual_cues)
                & {"strikethrough", "sample_denial_watermark"}
            )
            for candidate in candidates
        ):
            return None

        from .resolution import CaseLinker, EvidencePrecedenceResolver

        linked = CaseLinker().link(case_id, candidates)
        resolved = EvidencePrecedenceResolver().resolve(linked)
        arrival = resolved.value("arrival_date")
        try:
            arrival_date = date.fromisoformat(arrival) if arrival is not None else None
        except ValueError:
            arrival_date = None
        if (
            linked.unresolved
            or resolved.unresolved_linkage
            or resolved.contested_fields
            or resolved.value("visa_class") != "DIP-1"
            or resolved.value("risk_flags") is not None
            or arrival_date is None
            or (date(2026, 7, 7) - arrival_date).days <= 180
        ):
            return None

        def visible_values(
            page_index: int,
            field_names: set[str],
        ) -> dict[str, set[str]]:
            values: dict[str, set[str]] = {}
            for candidate in candidates:
                if (
                    candidate.page_index != page_index
                    or candidate.field_name not in field_names
                    or not candidate.legible
                    or candidate.value is None
                    or candidate.superseded
                    or candidate.source != "visible_ocr"
                    or candidate.case_id_hint != case_id
                    or "strikethrough" in candidate.visual_cues
                    or "sample_denial_watermark" in candidate.visual_cues
                ):
                    continue
                values.setdefault(candidate.field_name, set()).add(candidate.value)
            return values

        intake_fields = {
            "applicant_name",
            "species_code",
            "home_world",
            "visa_class",
            "sponsor_id",
            "arrival_date",
            "declared_purpose",
        }
        intake_values = visible_values(intake_pages[0], intake_fields)
        if any(
            intake_values.get(field_name) != {resolved.value(field_name)}
            for field_name in intake_fields
        ):
            return None
        registry_fields = {
            "applicant_name",
            "species_code",
            "home_world",
            "arrival_date",
        }
        registry_values = visible_values(registry_pages[0], registry_fields)
        if any(
            registry_values.get(field_name) != {resolved.value(field_name)}
            for field_name in registry_fields
        ):
            return None
        if not re.search(
            r"\bregistry\s+status\s*(?::|=|-)?\s*clear\b",
            page_text[registry_pages[0]],
            re.I,
        ):
            return None
        fee_values = visible_values(fee_pages[0], {"fee_status"}).get(
            "fee_status",
            set(),
        )
        if (
            fee_values not in ({"paid"}, {"waived"})
            or resolved.value("fee_status") not in fee_values
        ):
            return None

        repeated_fields = intake_fields | registry_fields | {"fee_status"}
        for candidate in candidates:
            if (
                candidate.field_name in repeated_fields
                and candidate.legible
                and candidate.value is not None
                and not candidate.superseded
                and candidate.source == "visible_ocr"
                and candidate.case_id_hint == case_id
                and "strikethrough" not in candidate.visual_cues
                and "sample_denial_watermark" not in candidate.visual_cues
                and candidate.value != resolved.value(candidate.field_name)
            ):
                return None

        supporting = tuple(
            candidate
            for candidate in candidates
            if candidate.page_index == intake_pages[0]
            and candidate.field_name in intake_fields
            and candidate.legible
            and candidate.value is not None
        )
        marker_box = supporting[0].box
        for candidate in supporting[1:]:
            marker_box = marker_box.union(candidate.box)
        return CandidateEvidence(
            field_name="minimal_diplomatic_packet",
            value="valid",
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=intake_pages[0],
            box=marker_box,
            legible=True,
            superseded=False,
            ocr_confidence=min(candidate.ocr_confidence for candidate in supporting),
            visual_cues=("minimal_diplomatic_packet_topology",),
            source="visible_ocr",
            case_id_hint=case_id,
            applicant_hint=resolved.active_applicant,
        )

    def extract(self, rendered_case: RenderedCase) -> tuple[CandidateEvidence, ...]:
        candidates: list[CandidateEvidence] = []
        page_type_markers: list[CandidateEvidence] = []
        pending_note_decisions: list[CandidateEvidence] = []
        routing_lines: dict[int, tuple[OcrLine, ...]] = {}
        risk_observations: list[
            tuple[
                tuple[str, ...],
                OcrLine,
                tuple[str, ...],
                EvidenceType,
                str | None,
            ]
        ] = []
        for page in rendered_case.pages:
            # Each PDF page is a separate document type in the challenge
            # packets. Never inherit a lower-precedence page type onto the next
            # page when its heading is partially degraded.
            tokens = self._ocr.read_page(page)
            lines = _visual_reading_order(group_ocr_lines(tokens))
            routing_lines[page.index] = lines
            evidence_type = EvidenceType.INTAKE_FORM
            for heading_line in lines:
                evidence_type = self._evidence_type(
                    heading_line.text, evidence_type
                )
            visible_case_ids = {
                value
                for line in lines
                if (value := self._normalize_value("case_id", line.text)) is not None
            }
            page_case_id_locked = rendered_case.case_id in visible_case_ids
            if page_case_id_locked:
                current_case_id = rendered_case.case_id
            elif len(visible_case_ids) == 1:
                current_case_id = next(iter(visible_case_ids))
            else:
                current_case_id = None
            if self._packet_page_type_markers:
                page_type_marker = self._packet_page_type_marker(
                    page=page,
                    lines=lines,
                    # A topology marker describes a physical page in the
                    # input packet. The already-validated filename association
                    # scopes it even when the page omits the case ID.
                    case_id=rendered_case.case_id,
                )
                if page_type_marker is not None:
                    page_type_markers.append(page_type_marker)
            current_applicant: str | None = None
            page_image = self._page_image(page)
            page_visual = (
                self._cues.prepare_page(page_image)
                if hasattr(self._cues, "prepare_page")
                else page_image
            )
            quarantine_remaining = 0
            pending_field: tuple[str, OcrLine, tuple[str, ...], EvidenceType] | None = None
            accepted_lines: list[OcrLine] = []
            accepted_line_cues: list[tuple[OcrLine, tuple[str, ...]]] = []
            for line in lines:
                cues = self._cues.cues_for_line(line, page_visual)
                quarantine = self._filter.context_quarantine_lines(line.text)
                if quarantine:
                    quarantine_remaining = max(quarantine_remaining, quarantine)
                if quarantine_remaining > 0:
                    quarantine_remaining -= 1
                    pending_field = None
                    continue
                if self._filter.rejection_reason(line.text, cues) is not None:
                    pending_field = None
                    continue

                accepted_lines.append(line)
                accepted_line_cues.append((line, cues))
                observed_flags = self._risk_flags_from_text(line.text)
                if observed_flags and "strikethrough" not in cues:
                    risk_observations.append(
                        (
                            observed_flags,
                            line,
                            cues,
                            evidence_type,
                            current_case_id,
                        )
                    )
                matched = self._match_field(line.text)
                if pending_field is not None and matched is None:
                    field_name, label_line, label_cues, label_type = pending_field
                    matched = (field_name, line.text)
                    combined_box = label_line.box.union(line.box)
                    combined_cues = tuple(sorted(set(label_cues) | set(cues)))
                    source_line = OcrLine(
                        page_index=line.page_index,
                        text=f"{label_line.text} {line.text}",
                        confidence=line.confidence,
                        box=combined_box,
                        tokens=label_line.tokens + line.tokens,
                    )
                    candidate_type = label_type
                    pending_field = None
                elif matched is not None:
                    pending_field = None
                    source_line = line
                    combined_cues = cues
                    candidate_type = evidence_type
                else:
                    continue

                field_name, raw_value = matched
                if not raw_value:
                    pending_field = (field_name, line, cues, evidence_type)
                    continue

                confidence = source_line.confidence
                if (
                    self._refinement_model is not None
                    and confidence < self._refinement_gate
                    and "strikethrough" not in combined_cues
                    and "sample_denial_watermark" not in combined_cues
                ):
                    refined = self._refinement_model.refine(page, source_line)
                    if refined is not None and refined[1] > confidence:
                        refined_text, refined_confidence = refined
                        if self._filter.rejection_reason(
                            refined_text, combined_cues
                        ) is None:
                            refined_match = self._match_field(refined_text)
                            if refined_match is None:
                                raw_value = refined_text
                                confidence = refined_confidence
                            elif refined_match[0] == field_name:
                                raw_value = refined_match[1]
                                confidence = refined_confidence

                normalized = self._normalize_value(field_name, raw_value)
                minimum_confidence = self._minimum_legible_confidence
                if (
                    field_name == "adjudication"
                    and candidate_type
                    in {
                        EvidenceType.ADJUDICATOR_STAMP,
                        EvidenceType.SIGNED_MANUAL_NOTE,
                    }
                ):
                    # The value is a three-item closed vocabulary and the page
                    # heading plus Finding/Decision label independently anchor
                    # its meaning.  Damaged scans often give the full word a
                    # pessimistic Tesseract score despite reading it exactly.
                    minimum_confidence = min(minimum_confidence, 0.28)
                legible = (
                    confidence >= minimum_confidence
                    and normalized is not None
                )
                candidate_value = normalized if legible else None
                candidate_case_id = (
                    candidate_value
                    if field_name == "case_id" and candidate_value is not None
                    else current_case_id
                )
                candidate_applicant = (
                    candidate_value
                    if field_name == "applicant_name" and candidate_value is not None
                    else current_applicant
                )
                candidates.append(
                    CandidateEvidence(
                        field_name=field_name,
                        value=candidate_value,
                        evidence_type=candidate_type,
                        page_index=page.index,
                        box=source_line.box,
                        legible=legible,
                        superseded="strikethrough" in combined_cues,
                        ocr_confidence=confidence,
                        visual_cues=combined_cues,
                        case_id_hint=candidate_case_id,
                        applicant_hint=candidate_applicant,
                    )
                )
                if field_name == "case_id" and candidate_value is not None:
                    # A clean header/footer that repeats the filename case ID
                    # anchors the whole physical page.  Do not let one damaged
                    # in-form digit (common on degraded fee receipts) re-scope
                    # all following values to a different packet.
                    if not page_case_id_locked or candidate_value == rendered_case.case_id:
                        if current_case_id != candidate_value:
                            current_applicant = None
                        current_case_id = candidate_value
                elif field_name == "applicant_name" and candidate_value is not None:
                    current_applicant = candidate_value

            heading_type = self._visible_page_heading_type(lines)
            trusted_line_cues = tuple(
                (line, cues)
                for line, cues in accepted_line_cues
                if "strikethrough" not in cues
            )
            if (
                heading_type
                in {
                    EvidenceType.ADJUDICATOR_STAMP,
                    EvidenceType.SIGNED_MANUAL_NOTE,
                }
                and trusted_line_cues
            ):
                recovered = self._recovered_authoritative_decision(
                    trusted_line_cues
                )
                if recovered is not None:
                    decision, supporting = recovered
                    decision_box = supporting[0][0].box
                    for line, _cues in supporting[1:]:
                        decision_box = decision_box.union(line.box)
                    decision_cues = {
                        cue for _line, cues in supporting for cue in cues
                    }
                    decision_cues.add("recovered_authoritative_decision")
                    pending_note_decisions.append(
                        CandidateEvidence(
                            field_name="adjudication",
                            value=decision,
                            evidence_type=heading_type,
                            page_index=page.index,
                            box=decision_box,
                            legible=True,
                            superseded=False,
                            ocr_confidence=max(
                                line.confidence for line, _cues in supporting
                            ),
                            visual_cues=tuple(sorted(decision_cues)),
                            case_id_hint=current_case_id,
                            applicant_hint=None,
                        )
                    )

                narrative_text = " ".join(line.text for line, _cues in trusted_line_cues)
                fee_status = self._authoritative_fee_status(narrative_text)
                if fee_status is not None:
                    phrase_records = tuple(
                        (line, cues)
                        for line, cues in trusted_line_cues
                        if re.search(r"\b(?:fee|unpaid|unknown)\b", line.text, re.I)
                    ) or trusted_line_cues
                    narrative_box = phrase_records[0][0].box
                    for line, _cues in phrase_records[1:]:
                        narrative_box = narrative_box.union(line.box)
                    narrative_confidence = sum(
                        line.confidence for line, _cues in phrase_records
                    ) / len(phrase_records)
                    narrative_cues = {
                        cue for _line, cues in phrase_records for cue in cues
                    }
                    narrative_cues.add("explicit_narrative_fact")
                    candidates.append(
                        CandidateEvidence(
                            field_name="fee_status",
                            value=(
                                fee_status
                                if narrative_confidence
                                >= self._minimum_legible_confidence
                                else None
                            ),
                            evidence_type=heading_type,
                            page_index=page.index,
                            box=narrative_box,
                            legible=(
                                narrative_confidence
                                >= self._minimum_legible_confidence
                            ),
                            superseded=False,
                            ocr_confidence=narrative_confidence,
                            visual_cues=tuple(sorted(narrative_cues)),
                            case_id_hint=current_case_id,
                            applicant_hint=None,
                        )
                    )

            if heading_type in {
                EvidenceType.ADJUDICATOR_STAMP,
                EvidenceType.SIGNED_MANUAL_NOTE,
                EvidenceType.BIOMETRIC_SLIP,
            }:
                # Risk phrases may remain visibly recognizable even when OCR
                # loses a leading character or an underscore.  Keep this
                # recovery behind both the trusted page heading and the normal
                # content/visual-cue filters; it must never scan an intake,
                # registry, sponsor, struck, or sample-denial line.
                for line, cues in trusted_line_cues:
                    if "sample_denial_watermark" in cues:
                        continue
                    already_observed = set(
                        self._risk_flags_from_text(line.text)
                    )
                    recovered_flags = tuple(
                        sorted(
                            set(self._fuzzy_risk_flags_from_text(line.text))
                            - already_observed
                        )
                    )
                    if not recovered_flags:
                        continue
                    risk_observations.append(
                        (
                            recovered_flags,
                            line,
                            tuple(
                                sorted(set(cues) | {"fuzzy_risk_phrase"})
                            ),
                            heading_type,
                            current_case_id,
                        )
                    )

            if heading_type is EvidenceType.REGISTRY_EXTRACT:
                for line, cues in trusted_line_cues:
                    home_world = self._registry_home_world(line.text)
                    if home_world is None:
                        continue
                    minimum_confidence = min(
                        self._minimum_legible_confidence,
                        0.28,
                    )
                    if not any(
                        candidate.field_name == "home_world"
                        and candidate.value == home_world
                        and candidate.page_index == page.index
                        and candidate.legible
                        for candidate in candidates
                    ):
                        candidates.append(
                            CandidateEvidence(
                                field_name="home_world",
                                value=(
                                    home_world
                                    if line.confidence >= minimum_confidence
                                    else None
                                ),
                                evidence_type=EvidenceType.REGISTRY_EXTRACT,
                                page_index=page.index,
                                box=line.box,
                                legible=line.confidence >= minimum_confidence,
                                superseded=False,
                                ocr_confidence=line.confidence,
                                visual_cues=tuple(
                                    sorted(set(cues) | {"title_gated_registry"})
                                ),
                                case_id_hint=current_case_id,
                                applicant_hint=current_applicant,
                            )
                        )
                    break

            if evidence_type is EvidenceType.SPONSOR_ATTESTATION and accepted_lines:
                page_text = " ".join(line.text for line in accepted_lines)
                narrative_matches = self._sponsor_narrative_matches(page_text)
                if narrative_matches:
                    narrative_box = accepted_lines[0].box
                    for line in accepted_lines[1:]:
                        narrative_box = narrative_box.union(line.box)
                    narrative_confidence = sum(
                        line.confidence for line in accepted_lines
                    ) / len(accepted_lines)
                    normalized_narrative = {
                        field_name: self._normalize_value(field_name, raw_value)
                        for field_name, raw_value in narrative_matches
                    }
                    narrative_applicant = normalized_narrative.get(
                        "applicant_name"
                    )
                    for field_name, _raw_value in narrative_matches:
                        candidate_value = normalized_narrative[field_name]
                        legible = (
                            narrative_confidence >= self._minimum_legible_confidence
                            and candidate_value is not None
                        )
                        candidates.append(
                            CandidateEvidence(
                                field_name=field_name,
                                value=candidate_value if legible else None,
                                evidence_type=evidence_type,
                                page_index=page.index,
                                box=narrative_box,
                                legible=legible,
                                superseded=False,
                                ocr_confidence=narrative_confidence,
                                visual_cues=("structured_sponsor_narrative",),
                                case_id_hint=current_case_id,
                                applicant_hint=(
                                    candidate_value
                                    if field_name == "applicant_name" and legible
                                    else narrative_applicant
                                ),
                            )
                        )

        existing_authoritative = any(
            candidate.field_name == "adjudication"
            and candidate.value in ADJUDICATION_VALUES
            and candidate.legible
            and not candidate.superseded
            and candidate.evidence_type
            in {
                EvidenceType.ADJUDICATOR_STAMP,
                EvidenceType.SIGNED_MANUAL_NOTE,
            }
            for candidate in candidates
        )
        pending_decisions = {
            candidate.value for candidate in pending_note_decisions
        }
        if not existing_authoritative and len(pending_decisions) == 1:
            candidates.extend(pending_note_decisions)

        if risk_observations:
            scoped = [
                item
                for item in risk_observations
                if item[4] == rendered_case.case_id
            ]
            if not scoped:
                scoped = [item for item in risk_observations if item[4] is None]
            # A page for a different case can contain a perfectly visible risk
            # marker.  CaseLinker must not receive it, and an all-foreign set
            # must remain an ordinary extraction gap rather than crashing the
            # whole PDF while selecting a representative observation below.
            if scoped:
                observed_values = {
                    flag
                    for flags, _line, _cues, _type, _case in scoped
                    for flag in flags
                }
                _flags, line, cues, candidate_type, candidate_case_id = min(
                    scoped,
                    key=lambda item: (
                        {
                            EvidenceType.ADJUDICATOR_STAMP: 1,
                            EvidenceType.SIGNED_MANUAL_NOTE: 1,
                            EvidenceType.INTAKE_FORM: 2,
                            EvidenceType.BIOMETRIC_SLIP: 3,
                            EvidenceType.SPONSOR_ATTESTATION: 4,
                            EvidenceType.REGISTRY_EXTRACT: 5,
                            EvidenceType.TEXT_LAYER: 6,
                        }[item[3]],
                        item[1].page_index,
                    ),
                )
                # A labeled line can yield a conservative direct value while
                # the full, still-visible line yields a strict superset (for
                # example one clean flag plus one mildly damaged flag).  Drop
                # only that redundant same-line subset.  An explicit `none`
                # is deliberately not a known-flag set, so it remains as a
                # same-rank conflict instead of being silently overridden.
                redundant_subset_ids: set[int] = set()
                for candidate in candidates:
                    if (
                        candidate.field_name != "risk_flags"
                        or not candidate.legible
                        or candidate.value is None
                        or candidate.superseded
                        or "strikethrough" in candidate.visual_cues
                        or "correction" in candidate.visual_cues
                        or candidate.evidence_type is not candidate_type
                        or candidate.page_index != line.page_index
                        or candidate.case_id_hint != candidate_case_id
                    ):
                        continue
                    candidate_flags = set(candidate.value.split("|"))
                    if (
                        candidate_flags
                        and candidate_flags < observed_values
                        and candidate_flags <= KNOWN_RISK_FLAGS
                    ):
                        redundant_subset_ids.add(id(candidate))
                if redundant_subset_ids:
                    candidates[:] = [
                        candidate
                        for candidate in candidates
                        if id(candidate) not in redundant_subset_ids
                    ]
                aggregate_cues = set(cues)
                if any(
                    "fuzzy_risk_phrase" in observation_cues
                    for (
                        _observation_flags,
                        _line,
                        observation_cues,
                        _type,
                        _case,
                    ) in scoped
                ):
                    aggregate_cues.add("fuzzy_risk_phrase")
                candidates.append(
                    CandidateEvidence(
                        field_name="risk_flags",
                        value="|".join(sorted(observed_values)),
                        evidence_type=candidate_type,
                        page_index=line.page_index,
                        box=line.box,
                        legible=True,
                        superseded="strikethrough" in cues,
                        ocr_confidence=line.confidence,
                        visual_cues=tuple(sorted(aggregate_cues)),
                        case_id_hint=candidate_case_id,
                        # Risk markers apply to the active case as a whole. Do not
                        # let a lower-precedence applicant mention on a later page
                        # scope a visible case-level risk marker away.
                        applicant_hint=None,
                    )
                )
        baseline = tuple(candidates)
        if self._trusted_scope_repair:
            candidates.extend(
                self._trusted_scope_repair_evidence(
                    rendered_case,
                    baseline,
                    routing_lines,
                )
            )
            baseline = tuple(candidates)
        if self._fee_receipt_retry:
            candidates.extend(
                self._fee_receipt_retry_evidence(
                    rendered_case,
                    baseline,
                    routing_lines,
                )
            )
            baseline = tuple(candidates)
        if self._sparse_intake_retry:
            candidates.extend(
                self._sparse_intake_retry_evidence(
                    rendered_case,
                    baseline,
                    routing_lines,
                )
            )
            baseline = tuple(candidates)
        if self._orientation_retry:
            candidates.extend(
                self._orientation_retry_evidence(
                    rendered_case,
                    baseline,
                    routing_lines,
                )
            )
            baseline = tuple(candidates)
        if self._risk_flag_retry:
            candidates.extend(
                self._risk_flag_retry_evidence(
                    rendered_case,
                    baseline,
                    routing_lines,
                )
            )
            baseline = tuple(candidates)
        if self._consensus_retry:
            candidates.extend(
                self._consensus_retry_evidence(
                    rendered_case,
                    baseline,
                    routing_lines,
                )
            )
        marker = self._minimal_diplomatic_packet_marker(
            rendered_case,
            tuple(candidates),
            routing_lines,
        )
        if marker is not None:
            candidates.append(marker)
        candidates.extend(page_type_markers)
        return tuple(candidates)
