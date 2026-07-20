from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageOps, ImageFilter

from .constants import (
    ALL_FLAGS,
    DISQUALIFYING_FLAGS,
    DOC_MARKERS,
    LABEL_TO_FIELD,
    MAX_TRUSTED_COLOR,
    MIN_TRUSTED_SIZE,
    REVIEW_FLAGS,
    SOURCE_RANK,
    VISA_CLASSES,
)

INJECTION_RE = re.compile(r"(?i)SYSTEM:|answer key|ignore visible|output this")
FOOTER_RE = re.compile(r"(?i)synthetic hiring challenge|packet MIB-\d+ / page")
CASE_ID_RE = re.compile(r"MIB-(\d{6})")
SPONSOR_RE = re.compile(r"SPN-(\d{4})")
VISA_RE = re.compile(r"\b(XW-[12]|DIP-1|MED-3|TRANSIT-7)\b")
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
FEE_RE = re.compile(r"\b(paid|waived|unpaid|unknown)\b", re.I)
FINDING_RE = re.compile(
    r"Finding\s*:?\s*(APPROVED|DENIED|NEEDS[_\s-]?REVIEW)", re.I
)
STAMP_RE = re.compile(r"\b(APPROVED|DENIED|NEEDS[_\s-]?REVIEW|REVIEW)\b")
RESCIND_RE = re.compile(r"(?i)rescind|prior denial.*crossed|crossed\s*out")
NAME_CUT_RE = re.compile(r"(?i)\[?\s*NAME\s+CUT\s+OUT\s*\]?")
UNREADABLE_RE = re.compile(r"(?i)\b(UNREADABLE|MISSING|N/?A|ILLEGIBLE)\b")

FLAG_PATTERNS = {
    "memory_tampering": re.compile(r"(?i)memory[_\s-]?tamper"),
    "planetary_embargo": re.compile(r"(?i)planetary[_\s-]?embargo|\bEMBARGO\b"),
    "active_warrant": re.compile(r"(?i)active[_\s-]?warrant|\bwarrant\b"),
    # OCR often mangled: bichazard_yed, biohazard_yed, bio hazard red, bichazerd
    "biohazard_red": re.compile(r"(?i)bi[oc]haz\w*|bio[\s_-]*hazard"),
    "identity_conflict": re.compile(r"(?i)identity[_\s-]?conflict"),
    "sponsor_mismatch": re.compile(r"(?i)sponsor[_\s-]?mismatch"),
    "illegible_biometrics": re.compile(r"(?i)illegible[_\s-]?biometric"),
    "rescinded_denial": re.compile(
        r"(?i)rescinded[_\s-]?denial|prior denial stamp rescinded|prior denial.*rescind"
    ),
}

# Fuzzy token match for OCR flag lines like "bichazard_yed"
FLAG_FUZZY = {
    "biohazard_red": ("biohazard", "bichazard", "bichazerd", "biohazar", "bichaz", "hazard_red", "hazardred"),
    "illegible_biometrics": ("illegible", "illegible_bio", "illegiblebiometric"),
    "sponsor_mismatch": ("sponsor_mismatch", "sponsormismatch"),
    "identity_conflict": ("identity_conflict", "identityconflict"),
    "rescinded_denial": ("rescinded", "rescind"),
    "memory_tampering": ("memory_tamper", "memorytamper"),
    "planetary_embargo": ("planetary_embargo", "embargo"),
    "active_warrant": ("active_warrant", "warrant"),
}

OCR_VISA_FIXES = {
    "DIPA": "DIP-1",
    "DIP": "DIP-1",
    "DIP1": "DIP-1",
    "XW1": "XW-1",
    "XW2": "XW-2",
    "XWE1": "XW-1",
    "XWE2": "XW-2",
    "XWI": "XW-1",
    "XWET": "XW-1",
    "MED3": "MED-3",
    "MED": "MED-3",
    "TRANSIT7": "TRANSIT-7",
    "TRANSIT": "TRANSIT-7",
}

OCR_FEE_FIXES = {
    "sumpaid": "unpaid",
    "umpaid": "unpaid",
    "unpaicl": "unpaid",
    "unpaic": "unpaid",
    "unkown": "unknown",
    "unkonwn": "unknown",
    "waivod": "waived",
    "waivcd": "waived",
    "unved": "waived",
    "waiv": "waived",
    "waved": "waived",
    "pac": "paid",
    "paid": "paid",
    "paicl": "paid",
    "waived": "waived",
    "unpaid": "unpaid",
    "unknown": "unknown",
}


@dataclass
class FieldHit:
    value: str
    source: str
    page: int
    confidence: float = 1.0


@dataclass
class PageExtract:
    page_index: int
    doc_type: str
    trusted_text: str
    spans: list[dict[str, Any]]
    fields: dict[str, FieldHit] = field(default_factory=dict)
    used_ocr: bool = False
    n_trusted_spans: int = 0


@dataclass
class PacketExtract:
    case_id: str
    pdf_path: str
    pages: list[PageExtract]
    fields: dict[str, FieldHit]
    docs_present: set[str]
    risk_flags: set[str]
    manual_finding: str | None
    conflicts: list[str]
    evidence_issues: list[str]
    used_ocr: bool
    trusted_span_count: int


def _normalize_label(text: str) -> str:
    return text.strip().rstrip(":").lower()


def _normalize_finding(text: str) -> str | None:
    t = text.upper().replace(" ", "_").replace("-", "_")
    if "NEEDS" in t and "REVIEW" in t:
        return "NEEDS_REVIEW"
    if t == "REVIEW":
        return "NEEDS_REVIEW"
    if t in {"APPROVED", "DENIED"}:
        return t
    return None


def _clean_value(value: str) -> str:
    value = value.strip().strip("|").strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _is_trusted_span(span: dict[str, Any]) -> bool:
    text = span.get("text") or ""
    if not text.strip():
        return False
    if span.get("color", 0) >= MAX_TRUSTED_COLOR:
        return False
    if float(span.get("size", 0)) < MIN_TRUSTED_SIZE:
        return False
    if INJECTION_RE.search(text):
        return False
    return True


def _collect_spans(page: fitz.Page) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append(
                    {
                        "text": span.get("text", ""),
                        "size": float(span.get("size", 0)),
                        "color": int(span.get("color", 0)),
                        "bbox": span.get("bbox"),
                    }
                )
    return spans


def _detect_doc_type(text: str) -> str:
    for key, marker in DOC_MARKERS.items():
        if marker in text:
            return key
    return "unknown"


def _pair_label_values(
    spans: list[dict[str, Any]], source: str, page_index: int
) -> dict[str, FieldHit]:
    hits: dict[str, FieldHit] = {}
    texts = [s["text"] for s in spans if _is_trusted_span(s)]
    for i, raw in enumerate(texts):
        label = _normalize_label(raw)
        field = LABEL_TO_FIELD.get(label)
        if not field:
            # Inline "Key: Value"
            if ":" in raw:
                left, right = raw.split(":", 1)
                field = LABEL_TO_FIELD.get(_normalize_label(left))
                if field and right.strip():
                    hits[field] = FieldHit(
                        value=_clean_value(right),
                        source=source,
                        page=page_index,
                    )
            continue
        if i + 1 >= len(texts):
            continue
        nxt = texts[i + 1]
        if _normalize_label(nxt) in LABEL_TO_FIELD:
            continue
        if FOOTER_RE.search(nxt):
            continue
        hits[field] = FieldHit(value=_clean_value(nxt), source=source, page=page_index)
    return hits


def _regex_fields(text: str, source: str, page_index: int) -> dict[str, FieldHit]:
    hits: dict[str, FieldHit] = {}
    patterns = {
        "applicant_name": r"(?:Applicant|Registry Name)\s*[:|]?\s*([^\n|]+)",
        "species_code": r"(?:Species Code|Species Match)\s*[:|]?\s*([A-Za-z0-9_]+)",
        "home_world": r"Home World\s*[:|]?\s*([^\n|]+)",
        "visa_class": r"Visa Class\s*[:|]?\s*([A-Za-z0-9\-]+)",
        "sponsor_id": r"Sponsor(?:\s*ID)?\s*[:|]?\s*(SPN-\d{4})",
        "arrival_date": r"Arrival Date\s*[:|]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|UNREADABLE|MISSING|N/?A|ILLEGIBLE)",
        "declared_purpose": r"Declared Purpose\s*[:|]?\s*([^\n|]+)",
        "fee_status": r"Fee\s*Sta[A-Za-z]*\s*[:.|]?\s*([A-Za-z]+)",
        "observed_flags": r"Observed flags\s*[:|]?\s*([^\n|]+)",
        "registry_status": r"Registry Status\s*[:|]?\s*([A-Za-z ]+)",
        "waiver_code": r"Waiver Code\s*[:|]?\s*(\S+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            hits[key] = FieldHit(
                value=_clean_value(m.group(1)), source=source, page=page_index, confidence=0.9
            )

    m = FINDING_RE.search(text)
    if m:
        finding = _normalize_finding(m.group(1))
        if finding:
            hits["manual_finding"] = FieldHit(
                value=finding, source=source, page=page_index, confidence=1.0
            )

    # Sponsor letter free text
    if "attests that" in text.lower():
        sm = SPONSOR_RE.search(text)
        if sm and "sponsor_id" not in hits:
            hits["sponsor_id"] = FieldHit(
                value=f"SPN-{sm.group(1)}", source=source, page=page_index, confidence=0.8
            )
        vm = VISA_RE.search(text)
        if vm and "visa_class" not in hits:
            hits["visa_class"] = FieldHit(
                value=vm.group(1), source=source, page=page_index, confidence=0.7
            )
        nm = re.search(r"attests that\s+([A-Z][A-Za-z\- ]+?)\s+is expected", text)
        if nm and "applicant_name" not in hits:
            hits["applicant_name"] = FieldHit(
                value=_clean_value(nm.group(1)), source=source, page=page_index, confidence=0.7
            )

    # Manual correction overrides
    cm = re.search(r"(?i)Manual correction:\s*sponsor is\s*(SPN-\d{4})", text)
    if cm:
        hits["sponsor_id"] = FieldHit(
            value=cm.group(1).upper(), source="adjudicator_note", page=page_index, confidence=1.0
        )

    return hits


def _normalize_field_value(key: str, value: str) -> str | None:
    value = _clean_value(value)
    if not value:
        return None
    if key == "visa_class":
        compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()
        if value.upper() in VISA_CLASSES:
            return value.upper()
        return OCR_VISA_FIXES.get(compact)
    if key == "sponsor_id":
        m = SPONSOR_RE.search(value.upper())
        return f"SPN-{m.group(1)}" if m else None
    if key == "fee_status":
        low = re.sub(r"[^a-z]", "", value.lower())
        fixed = OCR_FEE_FIXES.get(low)
        if fixed:
            return fixed
        # Prefix / contains matches for mangled OCR (Fee Sta: unved, pac, etc.)
        for cand, norm in (
            ("unpaid", "unpaid"),
            ("waiv", "waived"),
            ("unved", "waived"),
            ("paid", "paid"),
            ("pac", "paid"),
            ("unknown", "unknown"),
            ("unkown", "unknown"),
        ):
            if low.startswith(cand) or cand in low:
                return norm
        return None
    if key == "arrival_date":
        if UNREADABLE_RE.search(value):
            return "UNREADABLE"
        m = DATE_RE.search(value)
        return m.group(1) if m else None
    if key == "species_code":
        code = re.sub(r"[^A-Za-z0-9_]", "", value).upper()
        return code or None
    if key == "home_world":
        # OCR quirks
        value = value.replace("¢", "c").replace("Bamard", "Barnard").replace("Barard", "Barnard")
        return value
    if key == "applicant_name":
        if NAME_CUT_RE.search(value):
            return "[NAME CUT OUT]"
        if FOOTER_RE.search(value) or INJECTION_RE.search(value):
            return None
        value = re.sub(r"\bSCAN IMAGE\b", "", value, flags=re.I).strip(" '\"|")
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    if key in {"observed_flags", "registry_status", "declared_purpose", "waiver_code"}:
        return value
    return value


def _merge_hits(
    target: dict[str, FieldHit], incoming: dict[str, FieldHit], conflicts: list[str]
) -> None:
    for key, hit in incoming.items():
        norm = _normalize_field_value(key, hit.value) if key != "manual_finding" else hit.value
        if not norm:
            continue
        hit = FieldHit(value=norm, source=hit.source, page=hit.page, confidence=hit.confidence)
        if key not in target:
            target[key] = hit
            continue
        existing = target[key]
        if existing.value == hit.value:
            if SOURCE_RANK.get(hit.source, 99) < SOURCE_RANK.get(existing.source, 99):
                target[key] = hit
            elif hit.confidence > existing.confidence:
                target[key] = hit
            continue
        # Prefer higher-trust source; for equal trust prefer higher confidence / canonical enums
        hit_rank = SOURCE_RANK.get(hit.source, 99)
        exist_rank = SOURCE_RANK.get(existing.source, 99)
        if hit_rank < exist_rank:
            conflicts.append(f"{key}:{existing.value}->{hit.value}")
            target[key] = hit
        elif hit_rank == exist_rank:
            if key == "fee_status" and existing.value == "unknown" and hit.value != "unknown":
                target[key] = hit
            elif hit.confidence > existing.confidence + 0.05:
                conflicts.append(f"{key}_conflict:{existing.value}|{hit.value}")
                target[key] = hit
            else:
                conflicts.append(f"{key}_conflict:{existing.value}|{hit.value}")


def _fuzzy_flag_token(token: str) -> str | None:
    t = re.sub(r"[^a-z0-9_]", "", token.lower())
    if not t or t in {"none", "na", "n", "a"}:
        return None
    if t in ALL_FLAGS:
        return t
    for flag, needles in FLAG_FUZZY.items():
        for needle in needles:
            if needle in t or t in needle:
                return flag
    return None


def _flags_from_text(text: str) -> set[str]:
    found: set[str] = set()
    for name, pat in FLAG_PATTERNS.items():
        if pat.search(text):
            found.add(name)
    # observed flags line (tolerant of OCR: fiags/tlags/Ubserved)
    m = re.search(r"(?:Observed|Ubserved|Obsened)\s*f[li]ags\s*[:|]?\s*([^\n]+)", text, re.I)
    if m:
        raw = m.group(1).strip().lower()
        if "risk panel missing" in raw:
            found.add("illegible_biometrics")
        elif raw not in {"none", "n/a", "na", ""}:
            for part in re.split(r"[|,;/]+", raw):
                mapped = _fuzzy_flag_token(part.strip())
                if mapped:
                    found.add(mapped)
    if re.search(r"(?i)EMBARGO\s+REVIEW", text):
        found.add("planetary_embargo")
    if RESCIND_RE.search(text):
        found.add("rescinded_denial")
    return found


def _ocr_image(img: Image.Image, try_rotate: bool = False) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    chunks: list[str] = []
    w, h = img.size
    header = img.crop((0, 0, max(1, int(w * 0.85)), max(1, int(h * 0.35))))
    # Mid/lower band often holds "Observed flags" on biometric slips.
    mid = img.crop((0, int(h * 0.25), w, min(h, int(h * 0.75))))

    def _run(target: Image.Image, sharpen: bool = False) -> None:
        gray = ImageOps.autocontrast(ImageOps.grayscale(target))
        if sharpen:
            gray = gray.filter(ImageFilter.SHARPEN)
        try:
            chunks.append(pytesseract.image_to_string(gray, config="--psm 6"))
        except Exception:
            pass

    # Plain autocontrast preserves faint flag lines; sharpen helps titles/IDs.
    _run(header, sharpen=False)
    _run(img, sharpen=False)
    _run(mid, sharpen=False)
    _run(img, sharpen=True)
    if try_rotate:
        gray = ImageOps.autocontrast(ImageOps.grayscale(img))
        for angle in (90, 270):
            try:
                chunks.append(
                    pytesseract.image_to_string(gray.rotate(angle, expand=True), config="--psm 6")
                )
            except Exception:
                continue
    return "\n".join(chunks)


def _ocr_page(
    page: fitz.Page, doc: fitz.Document, dpi: int = 200, try_rotate: bool = False
) -> str:
    parts: list[str] = []
    embedded_ok = False
    for imginfo in page.get_images(full=True):
        xref = imginfo[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < 200 or pix.height < 200:
                continue
            # Skip tiny portrait/passport images
            if pix.width <= 640 and pix.height <= 640:
                continue
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            parts.append(_ocr_image(img, try_rotate=try_rotate))
            embedded_ok = True
        except Exception:
            continue
    # Fallback raster only when no usable embedded scan exists.
    if not embedded_ok:
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            parts.append(_ocr_image(img, try_rotate=try_rotate))
        except Exception:
            pass
    return "\n".join(p for p in parts if p.strip())


def _page_needs_ocr(page_extract: PageExtract, critical_missing: bool) -> bool:
    if page_extract.n_trusted_spans < 8:
        return True
    if critical_missing and page_extract.doc_type in {
        "intake",
        "fee_receipt",
        "biometric",
        "adjudicator_note",
        "unknown",
    }:
        return True
    return False


def extract_packet(pdf_path: Path, case_id: str | None = None) -> PacketExtract:
    pdf_path = Path(pdf_path)
    inferred_id = case_id or pdf_path.stem
    doc = fitz.open(pdf_path)

    pages: list[PageExtract] = []
    merged: dict[str, FieldHit] = {}
    conflicts: list[str] = []
    docs_present: set[str] = set()
    risk_flags: set[str] = set()
    used_ocr = False
    trusted_span_count = 0

    # Pass 1: trusted text layer
    for i, page in enumerate(doc):
        spans = _collect_spans(page)
        trusted = [s for s in spans if _is_trusted_span(s)]
        trusted_span_count += len(trusted)
        text = "\n".join(s["text"] for s in trusted)
        doc_type = _detect_doc_type(text)
        if doc_type != "unknown":
            docs_present.add(doc_type)
        pe = PageExtract(
            page_index=i,
            doc_type=doc_type,
            trusted_text=text,
            spans=trusted,
            n_trusted_spans=len(trusted),
        )
        hits = _pair_label_values(trusted, doc_type, i)
        _merge_hits(pe.fields, hits, [])
        regex_hits = _regex_fields(text, doc_type, i)
        _merge_hits(pe.fields, regex_hits, [])
        _merge_hits(merged, pe.fields, conflicts)
        risk_flags |= _flags_from_text(text)

        # Large stamp spans
        for s in trusted:
            if float(s.get("size", 0)) >= 18:
                finding = _normalize_finding(s["text"])
                if finding and "SAMPLE" not in s["text"].upper():
                    _merge_hits(
                        merged,
                        {
                            "manual_finding": FieldHit(
                                value=finding, source="adjudicator_note", page=i
                            )
                        },
                        conflicts,
                    )
        pages.append(pe)

    critical_keys = ["applicant_name", "visa_class", "fee_status", "arrival_date", "species_code"]
    critical_missing = any(k not in merged for k in critical_keys)
    image_heavy = trusted_span_count < 12

    # Pass 2: OCR when needed
    if critical_missing or image_heavy or "manual_finding" not in merged:
        for pe, page in zip(pages, doc):
            if not _page_needs_ocr(pe, critical_missing or image_heavy):
                # Still OCR adjudicator-looking image pages with almost no text
                if pe.n_trusted_spans >= 8:
                    continue
            ocr_text = _ocr_page(page, doc)
            if not ocr_text.strip():
                continue
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
            source = pe.doc_type if pe.doc_type != "unknown" else "ocr"
            # Re-detect doc type from OCR
            ocr_doc = _detect_doc_type(ocr_text)
            if ocr_doc != "unknown":
                pe.doc_type = ocr_doc
                docs_present.add(ocr_doc)
                source = ocr_doc
            hits = _regex_fields(ocr_text, source, pe.page_index)
            # Fee heuristic: amount line with $809 often means paid when status OCR fails
            if "fee_status" not in hits and re.search(r"\$\s*809", ocr_text):
                hits["fee_status"] = FieldHit(
                    value="paid", source=source, page=pe.page_index, confidence=0.55
                )
            if "fee_status" not in hits and re.search(r"\$\s*0\.00", ocr_text):
                hits["fee_status"] = FieldHit(
                    value="waived", source=source, page=pe.page_index, confidence=0.5
                )
            # Slightly lower confidence for OCR
            for h in hits.values():
                h.confidence *= 0.75
                h.source = source if source != "unknown" else "ocr"
            _merge_hits(pe.fields, hits, [])
            _merge_hits(merged, hits, conflicts)
            risk_flags |= _flags_from_text(ocr_text)
            fm = FINDING_RE.search(ocr_text)
            if fm:
                finding = _normalize_finding(fm.group(1))
                if finding:
                    _merge_hits(
                        merged,
                        {
                            "manual_finding": FieldHit(
                                value=finding,
                                source="adjudicator_note",
                                page=pe.page_index,
                                confidence=0.8,
                            )
                        },
                        conflicts,
                    )

    # Derive embargo flags from home world / registry even if not printed as risk flag
    home = merged.get("home_world")
    if home:
        if home.value in {"TRAPPIST-1e", "Eris Relay"}:
            risk_flags.add("planetary_embargo")
        if home.value == "Wolf-1061c":
            # Only a hard deny for non-DIP; still useful as signal
            pass
    reg = merged.get("registry_status")
    if reg and re.search(r"(?i)embargo", reg.value):
        risk_flags.add("planetary_embargo")

    # Observed flags field
    obs = merged.get("observed_flags")
    if obs:
        raw = obs.value.lower().strip()
        if "risk panel missing" in raw:
            risk_flags.add("illegible_biometrics")
        elif raw not in {"none", "n/a", "na", ""}:
            for part in re.split(r"[|,;/]+", raw):
                mapped = _fuzzy_flag_token(part.strip())
                if mapped:
                    risk_flags.add(mapped)
            # whole-string fuzzy for single mangled tokens
            mapped = _fuzzy_flag_token(raw)
            if mapped:
                risk_flags.add(mapped)

    evidence_issues: list[str] = []
    arrival = merged.get("arrival_date")
    if arrival and arrival.value == "UNREADABLE":
        evidence_issues.append("arrival_unreadable")
    if arrival is None:
        evidence_issues.append("arrival_missing")
    if "fee_status" not in merged:
        evidence_issues.append("fee_missing")
    if "visa_class" not in merged:
        evidence_issues.append("visa_missing")
    if "applicant_name" not in merged:
        evidence_issues.append("name_missing")
    name = merged.get("applicant_name")
    if name and name.value == "[NAME CUT OUT]":
        evidence_issues.append("name_cut_out")
        risk_flags.add("identity_conflict")
    if any(c.startswith("sponsor_id_conflict") or c.startswith("sponsor_id:") for c in conflicts):
        evidence_issues.append("sponsor_conflict")
        risk_flags.add("sponsor_mismatch")
    if image_heavy and critical_missing:
        evidence_issues.append("image_heavy_uncertain")

    # Name mismatch between intake and sponsor letter → sponsor_mismatch
    # Require substantial difference to avoid OCR false positives.
    all_text = "\n".join(p.trusted_text for p in pages)
    intake_names = re.findall(
        r"(?:^|\n)Applicant\s*[:|]?\s*([A-Z][A-Za-z\- ]{1,40})", all_text
    )
    letter_names = re.findall(
        r"attests that\s+([A-Z][A-Za-z\- ]+?)\s+is expected", all_text
    )
    if intake_names and letter_names:
        a = re.sub(r"[^a-z]", "", intake_names[0].lower())
        b = re.sub(r"[^a-z]", "", letter_names[0].lower())
        if a and b and "[namecutout]" not in a:
            # token sort ratio proxy: shared prefix/suffix length
            if a != b:
                shared = sum(1 for x, y in zip(a, b) if x == y)
                if shared < min(len(a), len(b)) * 0.6 and abs(len(a) - len(b)) + (
                    min(len(a), len(b)) - shared
                ) >= 4:
                    risk_flags.add("sponsor_mismatch")
                    evidence_issues.append("sponsor_name_mismatch")

    # Biometric illegible signal
    bio_text = "\n".join(p.trusted_text for p in pages if p.doc_type == "biometric")
    if bio_text and re.search(
        r"(?i)illegible|low confidence|confidence:\s*[0-4]\d%|RISK PANEL MISSING",
        bio_text,
    ):
        risk_flags.add("illegible_biometrics")

    manual_finding = None
    if "manual_finding" in merged:
        manual_finding = merged["manual_finding"].value

    # Case ID always from filename/caller — OCR/decoy case ids are untrusted.
    stem_match = CASE_ID_RE.search(pdf_path.stem) or CASE_ID_RE.search(case_id or "")
    inferred_id = f"MIB-{stem_match.group(1)}" if stem_match else (case_id or pdf_path.stem)

    return PacketExtract(
        case_id=inferred_id,
        pdf_path=str(pdf_path),
        pages=pages,
        fields=merged,
        docs_present=docs_present,
        risk_flags=risk_flags,
        manual_finding=manual_finding,
        conflicts=conflicts,
        evidence_issues=evidence_issues,
        used_ocr=used_ocr,
        trusted_span_count=trusted_span_count,
    )
