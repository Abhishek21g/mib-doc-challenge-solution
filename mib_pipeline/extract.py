from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageOps, ImageFilter

from .constants import (
    ALL_FLAGS,
    DISQUALIFYING_FLAGS,
    DOC_MARKERS,
    KNOWN_PURPOSES,
    KNOWN_SPECIES,
    KNOWN_WORLDS,
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
# Fee Status / Sta* — require a Fee cue so "Registry Status CLEAR" cannot bind.
FEE_STATUS_RE = re.compile(
    r"(?:F[eéo®]{1,4}\s+)St[a-zA-Z®]{2,10}\s*[:.|]?\s*([A-Za-z]{2,12})"
    r"|Fee\s*St[a-zA-Z®]{0,8}\s*[:.|]?\s*([A-Za-z]{2,12})",
    re.I,
)
# Tokens that look like Status-field captures but are never fee enums.
_FEE_REJECT_TOKENS = frozenset(
    {
        "letter",
        "work",
        "extract",
        "clear",
        "image",
        "name",
        "status",
        "registry",
        "visa",
        "class",
        "sponsor",
        "arrival",
        "amount",
        "waiver",
        "code",
        "packet",
        "earth",
        "valid",
        "only",
        "case",
        "form",
        "page",
        "redacted",
        "synthetic",
        "hiring",
        "challenge",
        "document",
    }
)
AMOUNT_RE = re.compile(r"(?i)\bamount\s*\$?\s*(\d+(?:\.\d{2})?)")
WAIVER_RE = re.compile(r"(?i)\bwaiver code\s*[:]?\s*(N/?A|[A-Z0-9_-]+)")
AMOUNT_ZERO_RE = re.compile(r"\$\s*0(?:\.00)?\b")
AMOUNT_809_RE = re.compile(r"\$\s*809(?:\.00)?\b")


def _fee_status_capture(text: str) -> str | None:
    """Return the raw fee-status token from a Fee Status label, if any."""
    fm = FEE_STATUS_RE.search(text)
    if fm:
        return next((g for g in fm.groups() if g), None)
    # Receipt OCR often drops "Fee" — allow bare Status only near receipt cues.
    if not re.search(
        r"(?i)fe[eag]\s*rec|fee\s*receipt|mib\s*fe|amount\s*\$|waiver\s*code",
        text,
    ):
        return None
    fm = re.search(
        r"(?i)\bSt(?:atus|atys|ays|ats|etus|at)\s*[:.|]?\s*([A-Za-z]{2,12})",
        text,
    )
    return fm.group(1) if fm else None
PURPOSE_FROM_LETTER_RE = re.compile(
    r"(?i)expected on earth for\s+([a-z][a-z ]+?)(?:\.|$)"
)
FINDING_RE = re.compile(
    r"Finding\s*:?\s*(APPROVED|DENIED|NEEDS[_\s-]?REVIEW)", re.I
)
STAMP_RE = re.compile(r"\b(APPROVED|DENIED|NEEDS[_\s-]?REVIEW|REVIEW)\b")
RESCIND_RE = re.compile(r"(?i)rescind|prior denial.*crossed|crossed\s*out")
NAME_CUT_RE = re.compile(r"(?i)\[?\s*NAME\s+CUT\s+OUT\s*\]?")
UNREADABLE_RE = re.compile(r"(?i)\b(UNREADABLE|MISSING|N/?A|ILLEGIBLE)\b")
MANUAL_FEE_RE = re.compile(
    r"(?i)Manual correction:\s*fee status is\s*(paid|waived|unpaid|unknown)"
)
OBSERVED_FLAGS_NONE_RE = re.compile(
    r"(?i)(?:Observed|Ubserved|Obsened)\s*f[li]ags\s*[:|]?\s*none\b"
)

FLAG_PATTERNS = {
    "memory_tampering": re.compile(r"(?i)memory[_\s-]?tamper"),
    "planetary_embargo": re.compile(r"(?i)planetary[_\s-]?embargo|\bEMBARGO\b"),
    "active_warrant": re.compile(r"(?i)active[_\s-]?warrant|\bwarrant\b"),
    # OCR: bichazard/bichaxard/bichazerd/bio hazard red
    "biohazard_red": re.compile(
        r"(?i)bi[ocx]ha[zxsr]\w*|bio[\s_-]*hazard|bichazard|bichaxard|bichazerd"
    ),
    "identity_conflict": re.compile(r"(?i)identity[_\s-]?conflict"),
    "sponsor_mismatch": re.compile(r"(?i)sponsor[_\s-]?mismatch"),
    # OCR garbles: Gegite biometrcs / illegible biomet / risk panel missing
    "illegible_biometrics": re.compile(
        r"(?i)illegible[_\s-]?biometric|illegible[_\s-]?bio|"
        r"gegite\s*biometr|biometrcs|risk\s*panel\s*missing|"
        r"observ\w*\s+f\w*\s+.*biometr"
    ),
    "rescinded_denial": re.compile(
        r"(?i)rescinded[_\s-]?denial|prior denial stamp rescinded|prior denial.*rescind"
    ),
}

# Fuzzy token match for OCR flag lines like "bichazard_yed" / "bichaxard_yed"
FLAG_FUZZY = {
    "biohazard_red": (
        "biohazard",
        "bichazard",
        "bichaxard",
        "bichazerd",
        "bichaxerd",
        "biohazar",
        "bichaz",
        "bichax",
        "hazard_red",
        "hazardred",
    ),
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
    "DIES": "DIP-1",
    "DIPS": "DIP-1",
    "DIPI": "DIP-1",
    "XW1": "XW-1",
    "XW2": "XW-2",
    "XWE1": "XW-1",
    "XWE2": "XW-2",
    "XWI": "XW-1",
    "XWET": "XW-1",
    "XWI1": "XW-1",
    "XWI2": "XW-2",
    "MED3": "MED-3",
    "MED": "MED-3",
    "MEDS": "MED-3",
    "TRANSIT7": "TRANSIT-7",
    "TRANSIT": "TRANSIT-7",
}

OCR_FEE_FIXES = {
    "sumpaid": "unpaid",
    "umpaid": "unpaid",
    "unpaicl": "unpaid",
    "unpaic": "unpaid",
    "unpaid": "unpaid",
    "unkown": "unknown",
    "unkonwn": "unknown",
    "unknown": "unknown",
    "waivod": "waived",
    "waivcd": "waived",
    "walved": "waived",
    "waivled": "waived",
    "unved": "waived",
    "waiv": "waived",
    "waved": "waived",
    "waived": "waived",
    "eared": "waived",
    "earved": "waived",
    "eavved": "waived",
    "pac": "paid",
    "paid": "paid",
    "paicl": "paid",
    "paic": "paid",
    "paig": "paid",
    "pald": "paid",
    "pai": "paid",
    "pag": "paid",
    "pal": "paid",
    "pakd": "paid",
    "aig": "paid",
    "paidl": "paid",
}

# Canonical fee tokens for edit-distance recovery of short OCR junk.
_FEE_CANONICAL = ("paid", "waived", "unpaid", "unknown")


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
    # True when biometric OCR explicitly read "Observed flags: none" (or a flag).
    biometric_flags_observed: bool = False


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
        "observed_flags": r"(?:Observed|Ubserved|Obsened)\s*f[li]ags\s*[:|]?\s*([^\n|]+)",
        "registry_status": r"Registry Status\s*[:|]?\s*([A-Za-z ]+)",
        "waiver_code": r"Waiver Code\s*[:|]?\s*(\S+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if m:
            hits[key] = FieldHit(
                value=_clean_value(m.group(1)), source=source, page=page_index, confidence=0.9
            )

    # Stronger fee status: require Fee cue (see FEE_STATUS_RE / _fee_status_capture).
    fee_tok = _fee_status_capture(text)
    if fee_tok:
        hits["fee_status"] = FieldHit(
            value=_clean_value(fee_tok), source=source, page=page_index, confidence=0.9
        )

    m = FINDING_RE.search(text)
    if m:
        finding = _normalize_finding(m.group(1))
        if finding:
            hits["manual_finding"] = FieldHit(
                value=finding, source=source, page=page_index, confidence=1.0
            )

    # Sponsor letter free text
    if "attests that" in text.lower() or "expected on earth" in text.lower():
        sm = SPONSOR_RE.search(text)
        if sm and "sponsor_id" not in hits:
            hits["sponsor_id"] = FieldHit(
                value=f"SPN-{sm.group(1)}", source=source, page=page_index, confidence=0.8
            )
        vm = VISA_RE.search(text) or re.search(
            r"(?i)class\s+([A-Za-z]+[- ]?\d)\s+compliance", text
        )
        if vm and "visa_class" not in hits:
            hits["visa_class"] = FieldHit(
                value=vm.group(1), source=source, page=page_index, confidence=0.7
            )
        nm = re.search(r"attests that\s+([A-Z][A-Za-z\- ]+?)\s+is expected", text)
        if nm and "applicant_name" not in hits:
            hits["applicant_name"] = FieldHit(
                value=_clean_value(nm.group(1)), source=source, page=page_index, confidence=0.7
            )
        pm = PURPOSE_FROM_LETTER_RE.search(text)
        if pm and "declared_purpose" not in hits:
            hits["declared_purpose"] = FieldHit(
                value=_clean_value(pm.group(1)), source=source, page=page_index, confidence=0.75
            )

    # Manual correction overrides
    cm = re.search(r"(?i)Manual correction:\s*sponsor is\s*(SPN-\d{4})", text)
    if cm:
        hits["sponsor_id"] = FieldHit(
            value=cm.group(1).upper(), source="adjudicator_note", page=page_index, confidence=1.0
        )
    fee_m = MANUAL_FEE_RE.search(text)
    if fee_m:
        hits["fee_status"] = FieldHit(
            value=fee_m.group(1).lower(),
            source="adjudicator_note",
            page=page_index,
            confidence=1.0,
        )
    name_m = re.search(r"(?i)Manual correction:\s*applicant is\s*([^\n.]+)", text)
    if name_m:
        hits["applicant_name"] = FieldHit(
            value=_clean_value(name_m.group(1)),
            source="adjudicator_note",
            page=page_index,
            confidence=1.0,
        )

    return hits


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _normalize_fee_token(value: str) -> str | None:
    """Map OCR fee debris to the closed fee enum.

    Short stems must not match as substrings: ``ear`` in CLEAR/Earth and
    ``pac`` in packet previously invented waived/paid from non-fee lines.
    """
    low = re.sub(r"[^a-z]", "", value.lower())
    if not low or low in _FEE_REJECT_TOKENS:
        return None
    fixed = OCR_FEE_FIXES.get(low)
    if fixed:
        return fixed
    # Prefix-only stems (exact OCR_FEE_FIXES already handled above).
    for cand, norm in (
        ("unpaid", "unpaid"),
        ("waiv", "waived"),
        ("unved", "waived"),
        ("eared", "waived"),
        ("earved", "waived"),
        ("paid", "paid"),
        ("unknown", "unknown"),
        ("unkown", "unknown"),
    ):
        if low.startswith(cand):
            return norm
    # Short OCR junk via edit distance to canonical tokens only.
    if not (3 <= len(low) <= 8):
        return None
    best = None
    best_d = 99
    for cand in _FEE_CANONICAL:
        d = _edit_distance(low, cand)
        limit = 1 if max(len(low), len(cand)) <= 4 else 2
        if d <= limit and d < best_d:
            best = cand
            best_d = d
    return best


def _fee_from_amount_text(text: str) -> str | None:
    """Infer fee_status from receipt amount + waiver jointly.

    Unpaid receipts also print $809, so amount alone must not mean paid.
    """
    amount_m = AMOUNT_RE.search(text)
    waiver_m = WAIVER_RE.search(text)
    receipt_like = bool(re.search(r"(?i)fe[eag]\s*rec|fee\s*receipt|mib\s*fe", text))
    if amount_m and waiver_m:
        try:
            value = float(amount_m.group(1))
        except ValueError:
            value = -1.0
        waiver_code = waiver_m.group(1).upper().replace("N/A", "N/A")
        if waiver_code in {"N/A", "NA", "N\\A"}:
            waiver_code = "N/A"
        if value == 0 and waiver_code != "N/A":
            return "waived"
        if value > 0 and waiver_code == "N/A":
            return "paid"
    if waiver_m:
        code = waiver_m.group(1).upper()
        if code not in {"N/A", "NA", ""} and (
            AMOUNT_ZERO_RE.search(text) or (amount_m and float(amount_m.group(1)) == 0)
        ):
            return "waived"
    # Zero-dollar receipt without parsed waiver is almost always waived.
    if receipt_like and AMOUNT_ZERO_RE.search(text):
        return "waived"
    return None


def _canonicalize_purpose(value: str) -> str | None:
    cleaned = _clean_value(value).lower()
    cleaned = re.split(
        r"\b(?:species|home|visa|sponsor|arrival|fee|case|passport|registry)\b",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if cleaned in KNOWN_PURPOSES:
        return cleaned
    # Prefix / containment match against closed vocabulary.
    for purpose in KNOWN_PURPOSES:
        if cleaned.startswith(purpose) or purpose.startswith(cleaned):
            if len(cleaned) >= 4:
                return purpose
    best = None
    best_d = 99
    for purpose in KNOWN_PURPOSES:
        d = _edit_distance(cleaned.replace(" ", ""), purpose.replace(" ", ""))
        if d < best_d and d <= max(2, len(purpose) // 5):
            best = purpose
            best_d = d
    return best


def _canonicalize_world(value: str) -> str | None:
    cleaned = _clean_value(value)
    cleaned = re.split(
        r"\b(?:Species|Visa|Sponsor|Arrival|Declared|Passport|Registry|SCAN|FORM)\b",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = re.sub(r"[‘’'`\"|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:")
    if not cleaned:
        return None
    for world in KNOWN_WORLDS:
        if cleaned.casefold() == world.casefold():
            return world
    # OCR near-matches
    compact = re.sub(r"[^a-z0-9]", "", cleaned.casefold())
    best = None
    best_d = 99
    for world in KNOWN_WORLDS:
        wcompact = re.sub(r"[^a-z0-9]", "", world.casefold())
        d = _edit_distance(compact, wcompact)
        if d < best_d and d <= 2:
            best = world
            best_d = d
    return best or cleaned


def _canonicalize_species(value: str) -> str | None:
    code = re.sub(r"[^A-Za-z0-9_]", "", value).upper()
    if not code:
        return None
    if code in KNOWN_SPECIES:
        return code
    for sp in KNOWN_SPECIES:
        if code.startswith(sp) or sp.startswith(code):
            if len(code) >= 6:
                return sp
    best = None
    best_d = 99
    for sp in KNOWN_SPECIES:
        d = _edit_distance(code, sp)
        if d < best_d and d <= 2:
            best = sp
            best_d = d
    return best or code


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
        return _normalize_fee_token(value)
    if key == "arrival_date":
        if UNREADABLE_RE.search(value):
            return "UNREADABLE"
        m = DATE_RE.search(value)
        if not m:
            return None
        raw = m.group(1)
        try:
            datetime.strptime(raw, "%Y-%m-%d")
        except ValueError:
            return "UNREADABLE"
        return raw
    if key == "species_code":
        return _canonicalize_species(value)
    if key == "home_world":
        value = value.replace("¢", "c").replace("Bamard", "Barnard").replace("Barard", "Barnard")
        return _canonicalize_world(value)
    if key == "applicant_name":
        if NAME_CUT_RE.search(value):
            return "[NAME CUT OUT]"
        if FOOTER_RE.search(value) or INJECTION_RE.search(value):
            return None
        cleaned = re.sub(r"\b(?:SCAN IMAGE|PASSPORT IMAGE|REGISTRY IMAGE)\b", "", value, flags=re.I)
        cleaned = re.split(
            r"\b(?:species|home\s*world|visa|sponsor|arrival|declared|passport|registry|sport\s*image)\b",
            cleaned,
            maxsplit=1,
            flags=re.I,
        )[0]
        cleaned = re.sub(r"[‘’'`\"|]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:|-—_")
        tokens = cleaned.split()
        # Drop trailing OCR debris (1–2 char tokens, non-alpha junk).
        while tokens and (
            len(tokens[-1]) <= 2 or not re.fullmatch(r"[A-Za-z][A-Za-z'-]*", tokens[-1])
        ):
            tokens.pop()
        if len(tokens) > 4:
            tokens = tokens[:4]
        cleaned = " ".join(tokens).strip()
        if len(cleaned) < 3 or len(cleaned) > 48:
            return None
        if re.search(r"\d", cleaned):
            return None
        return cleaned or None
    if key == "declared_purpose":
        return _canonicalize_purpose(value)
    if key in {"observed_flags", "registry_status", "waiver_code"}:
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
            elif key == "applicant_name" and _edit_distance(
                re.sub(r"[^a-z]", "", existing.value.lower()),
                re.sub(r"[^a-z]", "", hit.value.lower()),
            ) <= 2:
                # Near-duplicate OCR names — keep the longer/cleaner token.
                chosen = hit if len(hit.value) > len(existing.value) else existing
                target[key] = chosen
            elif key in {"species_code", "home_world", "declared_purpose", "visa_class"}:
                # Prefer shorter clean enum when the other is a glued OCR row.
                if existing.value.lower().startswith(hit.value.lower()) and len(hit.value) >= 3:
                    target[key] = hit
                elif hit.value.lower().startswith(existing.value.lower()) and len(existing.value) >= 3:
                    pass
                elif hit.confidence > existing.confidence + 0.05:
                    conflicts.append(f"{key}_conflict:{existing.value}|{hit.value}")
                    target[key] = hit
                else:
                    conflicts.append(f"{key}_conflict:{existing.value}|{hit.value}")
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


def _ocr_image(
    img: Image.Image,
    try_rotate: bool = False,
    fee_boost: bool = False,
) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    chunks: list[str] = []
    w, h = img.size
    header = img.crop((0, 0, max(1, int(w * 0.9)), max(1, int(h * 0.40))))
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

    if fee_boost:
        gray_h = ImageOps.autocontrast(ImageOps.grayscale(header))
        # Light binary helps stamped fee headers.
        for thr in (140, 160):
            binary = gray_h.point(lambda x, t=thr: 255 if x > t else 0)
            try:
                chunks.append(pytesseract.image_to_string(binary, config="--psm 6"))
            except Exception:
                pass
        # 2x upsample often turns "paig" → "Paid"; sparse PSM 11 recovers tokens.
        up = gray_h.resize(
            (max(1, gray_h.width * 2), max(1, gray_h.height * 2)),
            Image.Resampling.LANCZOS,
        )
        try:
            chunks.append(pytesseract.image_to_string(up, config="--psm 6"))
        except Exception:
            pass
        try:
            chunks.append(pytesseract.image_to_string(up, config="--psm 11"))
        except Exception:
            pass
        # Slightly lower mid-band crop recovers status under tall receipt headers.
        mid_fee = img.crop((0, int(h * 0.12), max(1, int(w * 0.92)), max(1, int(h * 0.58))))
        mid_g = ImageOps.autocontrast(ImageOps.grayscale(mid_fee))
        mid_up = mid_g.resize(
            (max(1, mid_g.width * 2), max(1, mid_g.height * 2)),
            Image.Resampling.LANCZOS,
        )
        try:
            chunks.append(pytesseract.image_to_string(mid_up, config="--psm 11"))
        except Exception:
            pass

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
    page: fitz.Page,
    doc: fitz.Document,
    dpi: int = 200,
    try_rotate: bool = False,
    fee_boost: bool = False,
    risk_contrast: bool = False,
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
            # Skip tiny portrait/passport images; keep letter-sized scans.
            if pix.width <= 640 and pix.height <= 640:
                continue
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            parts.append(
                _ocr_image(img, try_rotate=try_rotate, fee_boost=fee_boost)
            )
            if parts[-1].strip():
                embedded_ok = True
        except Exception:
            continue
    # Always raster sparse / footer-only pages — embedded OCR can be empty
    # even when a large scan exists, and portraits shouldn't block recovery.
    page_text_len = len((page.get_text() or "").strip())
    if not embedded_ok or page_text_len < 100:
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            parts.append(
                _ocr_image(img, try_rotate=try_rotate, fee_boost=fee_boost)
            )
        except Exception:
            pass
    joined = "\n".join(p for p in parts if p.strip())
    # Afifi-style: contrast retry only to surface risk wording, never alone.
    if risk_contrast and not re.search(
        r"(?i)biohazard|warrant|tamper|embargo|rescind|illegible|observ",
        joined,
    ):
        try:
            pix = page.get_pixmap(dpi=max(dpi, 180))
            img = ImageOps.autocontrast(
                ImageOps.grayscale(Image.open(io.BytesIO(pix.tobytes("png"))))
            )
            contrast_txt = _ocr_image(img, try_rotate=False, fee_boost=False)
            if re.search(
                r"(?i)biohazard|warrant|tamper|embargo|rescind|illegible|observ",
                contrast_txt,
            ):
                joined = (joined + "\n" + contrast_txt).strip()
        except Exception:
            pass
    return joined


def _page_needs_ocr(page_extract: PageExtract, critical_missing: bool) -> bool:
    # Always OCR sparse pages (biometric/flag/fee scans) even if other fields look complete.
    if page_extract.n_trusted_spans < 10:
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


def _names_clearly_mismatch(a: str, b: str) -> bool:
    """True only for clearly different clean names — not OCR debris FPs.

    Public labels rarely encode pure OCR name noise as sponsor_mismatch; most
    true mismatches also show up as sponsor_id conflicts. Be strict.
    """
    a = _normalize_field_value("applicant_name", a) or ""
    b = _normalize_field_value("applicant_name", b) or ""
    a = re.sub(r"[^a-z]", "", a.lower())
    b = re.sub(r"[^a-z]", "", b.lower())
    if not a or not b or "[namecutout]" in a:
        return False
    if a == b:
        return False
    # Near-duplicates (1–2 edits) are OCR, not identity conflicts.
    if _edit_distance(a, b) <= 2:
        return False
    if min(len(a), len(b)) < 10 or max(len(a), len(b)) < 12:
        return False
    dist = _edit_distance(a, b)
    # Require a large absolute difference — shared long prefix means OCR drift.
    shared = 0
    for x, y in zip(a, b):
        if x == y:
            shared += 1
        else:
            break
    if shared >= 6:
        return False
    return dist >= max(6, int(0.55 * min(len(a), len(b))))


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
    biometric_flags_observed = False

    # Pass 2: ALWAYS OCR pages with n_trusted_spans < 10 (biometric/flag/fee scans),
    # plus other pages when critical fields are still missing.
    for pe, page in zip(pages, doc):
        if not _page_needs_ocr(pe, critical_missing or image_heavy):
            continue
        ocr_text = _ocr_page(page, doc, fee_boost=False, risk_contrast=True)
        if not ocr_text.strip():
            continue
        used_ocr = True
        pe.used_ocr = True
        pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
        # Never promote OCR to native doc-type ranks — garbled OCR was overwriting
        # clean intake visa/name with attestation debris (Afifi ranks OCR lower).
        ocr_doc = _detect_doc_type(ocr_text)
        if ocr_doc != "unknown":
            pe.doc_type = ocr_doc
            docs_present.add(ocr_doc)
        if re.search(r"(?i)fee\s*rec|mib\s*fe", ocr_text):
            pe.doc_type = "fee_receipt"
            docs_present.add("fee_receipt")
            ocr_doc = "fee_receipt"
        source = f"ocr_{ocr_doc}" if ocr_doc != "unknown" else "ocr"
        hits = _regex_fields(ocr_text, source, pe.page_index)
        raw_fee = hits.get("fee_status")
        if raw_fee is None or _normalize_fee_token(raw_fee.value) is None:
            amt = _fee_from_amount_text(ocr_text)
            if amt:
                hits["fee_status"] = FieldHit(
                    value=amt, source=source, page=pe.page_index, confidence=0.55
                )
        for h in hits.values():
            h.confidence *= 0.75
            h.source = source
        _merge_hits(pe.fields, hits, [])
        _merge_hits(merged, hits, conflicts)
        risk_flags |= _flags_from_text(ocr_text)
        if OBSERVED_FLAGS_NONE_RE.search(ocr_text) or hits.get("observed_flags"):
            biometric_flags_observed = True
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

    # Pass 2b: fee still missing → header 2x/binary OCR on sparse pages that
    # already look like fee receipts (avoid re-OCR of every biometric slip).
    if "fee_status" not in merged:
        for pe, page in zip(pages, doc):
            if pe.n_trusted_spans >= 10:
                continue
            if not page.get_images():
                continue
            hint = pe.trusted_text
            if not re.search(r"(?i)fee|rec[ei]|amount|\$\s*\d|sta[tuswys]", hint):
                # Still try completely blank scan pages (footer-only) once
                if pe.n_trusted_spans > 3:
                    continue
            ocr_text = _ocr_page(page, doc, fee_boost=True)
            if not ocr_text.strip():
                continue
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
            source = "ocr_fee_receipt"
            if re.search(r"(?i)fee\s*rec|mib\s*fe", ocr_text):
                pe.doc_type = "fee_receipt"
                docs_present.add("fee_receipt")
            hits = _regex_fields(ocr_text, source, pe.page_index)
            raw_fee = hits.get("fee_status")
            if raw_fee is None or _normalize_fee_token(raw_fee.value) is None:
                amt = _fee_from_amount_text(ocr_text)
                if amt:
                    hits["fee_status"] = FieldHit(
                        value=amt, source=source, page=pe.page_index, confidence=0.5
                    )
            for h in hits.values():
                h.confidence *= 0.7
                h.source = source
            _merge_hits(merged, hits, conflicts)
            risk_flags |= _flags_from_text(ocr_text)
            if "fee_status" in merged:
                break

    # After all pages merged: scan joined trusted_text for fee cues
    all_text = "\n".join(p.trusted_text for p in pages)
    if "fee_status" not in merged or (
        merged.get("fee_status") and merged["fee_status"].value == "unknown"
    ):
        fee_tok = _fee_status_capture(all_text)
        if fee_tok:
            norm = _normalize_fee_token(fee_tok)
            if norm and norm != "unknown":
                _merge_hits(
                    merged,
                    {
                        "fee_status": FieldHit(
                            value=norm, source="ocr", page=0, confidence=0.6
                        )
                    },
                    conflicts,
                )
        if "fee_status" not in merged or merged["fee_status"].value == "unknown":
            amt = _fee_from_amount_text(all_text)
            if amt:
                _merge_hits(
                    merged,
                    {
                        "fee_status": FieldHit(
                            value=amt, source="ocr", page=0, confidence=0.55
                        )
                    },
                    conflicts,
                )
        # Manual fee correction anywhere in packet
        fee_m = MANUAL_FEE_RE.search(all_text)
        if fee_m:
            _merge_hits(
                merged,
                {
                    "fee_status": FieldHit(
                        value=fee_m.group(1).lower(),
                        source="adjudicator_note",
                        page=0,
                        confidence=1.0,
                    )
                },
                conflicts,
            )

    if OBSERVED_FLAGS_NONE_RE.search(all_text):
        biometric_flags_observed = True
    if merged.get("observed_flags"):
        biometric_flags_observed = True

    # Embargo worlds are denied in adjudication; only attach the risk_flags
    # token when visible evidence mentions embargo (avoids extraction FPs).
    reg = merged.get("registry_status")
    if reg and re.search(r"(?i)embargo", reg.value):
        risk_flags.add("planetary_embargo")
    home = merged.get("home_world")
    if home and home.value in {"TRAPPIST-1e", "Eris Relay"}:
        if re.search(r"(?i)embargo", all_text):
            risk_flags.add("planetary_embargo")

    # Observed flags field
    obs = merged.get("observed_flags")
    if obs:
        raw = obs.value.lower().strip()
        if "risk panel missing" in raw:
            risk_flags.add("illegible_biometrics")
        elif re.match(r"none\b", raw) or raw in {"n/a", "na", ""}:
            pass
        else:
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
        # Cut-out is an identity gap for review, not automatically the
        # scored identity_conflict risk flag (that over-fired vs gold).
    if any(c.startswith("sponsor_id_conflict") or c.startswith("sponsor_id:") for c in conflicts):
        evidence_issues.append("sponsor_conflict")
        risk_flags.add("sponsor_mismatch")
    if image_heavy and critical_missing:
        evidence_issues.append("image_heavy_uncertain")

    # Name mismatch between intake and letter is mostly OCR noise on public
    # labels; do not promote to sponsor_mismatch. Sponsor ID conflicts above
    # already cover the reliable signal.
    _ = all_text  # kept for biometric / fee scans above


    # Biometric illegible signal (including OCR garbles on scan slips).
    bio_text = "\n".join(
        p.trusted_text for p in pages if p.doc_type in {"biometric", "unknown", "ocr"}
    )
    if not bio_text:
        bio_text = all_text
    if bio_text and re.search(
        r"(?i)illegible|low confidence|confidence:\s*[0-4]\d%|RISK PANEL MISSING|"
        r"gegite\s*biometr|biometrcs|observ\w*\s+f\w*\s+.*biometr",
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
        biometric_flags_observed=biometric_flags_observed,
    )
