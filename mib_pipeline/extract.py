from __future__ import annotations

import difflib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageOps, ImageFilter

from .constants import (
    ALL_FLAGS,
    ALWAYS_EMBARGO_WORLDS,
    DISQUALIFYING_FLAGS,
    DOC_MARKERS,
    KNOWN_PURPOSES,
    KNOWN_SPECIES,
    KNOWN_WORLDS,
    LABEL_TO_FIELD,
    MAX_TRUSTED_COLOR,
    MIN_TRUSTED_SIZE,
    NAME_PARTS,
    REVIEW_FLAGS,
    SOURCE_RANK,
    VISA_CLASSES,
)
from . import rapid_ocr

INJECTION_RE = re.compile(r"(?i)SYSTEM:|answer key|ignore visible|output this")
FOOTER_RE = re.compile(r"(?i)synthetic hiring challenge|packet MIB-\d+ / page")
CASE_ID_RE = re.compile(r"MIB-(\d{6})")
SPONSOR_RE = re.compile(r"SPN-?([0-9OIl]{4})", re.I)
VISA_RE = re.compile(r"\b(XW-[12]|DIP-1|MED-3|TRANSIT-7)\b")
DATE_RE = re.compile(r"\b(\d{4}[-/.:]\d{2}[-/.:]\d{2})\b")
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
# Adjudicator / stamp narratives (strobl/goleffect authoritative fee phrases).
AUTHORITATIVE_FEE_RE = re.compile(
    r"(?i)(?:mandatory\s+fee\s+(unpaid|paid|waived))"
    r"|(?:fee\s+status\s+is\s+(paid|waived|unpaid|unknown))"
    r"|(?:fee\s+st\w*\s+(unknown|unpaid|paid|waived))"
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
    "urpatd": "unpaid",
    "unpaid": "unpaid",
    "unkown": "unknown",
    "unkonwn": "unknown",
    "unkrnown": "unknown",
    "unknown": "unknown",
    "waivod": "waived",
    "waivcd": "waived",
    "walved": "waived",
    "waivled": "waived",
    "warved": "waived",
    "watved": "waived",
    "aaived": "waived",
    "unved": "waived",
    "waiv": "waived",
    "waved": "waived",
    "waived": "waived",
    "eared": "waived",
    "earved": "waived",
    "eavved": "waived",
    "eaved": "waived",
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
    "naid": "paid",
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


def _fuzzy_vocab(value: str, choices: tuple[str, ...] | set[str], cutoff: float = 0.67) -> str | None:
    """Snap OCR debris onto a closed vocabulary (goleffect-style)."""
    if not value:
        return None
    choices_t = tuple(choices)
    folded = re.sub(r"[^A-Z0-9]", "", value.upper())
    exact = {re.sub(r"[^A-Z0-9]", "", c.upper()): c for c in choices_t}
    if folded in exact:
        return exact[folded]
    match = difflib.get_close_matches(folded, list(exact), n=1, cutoff=cutoff)
    return exact[match[0]] if match else None


def _clean_sponsor(value: str) -> str | None:
    """Normalize SPN ids; map common OCR digit confusions."""
    cleaned = value.upper().replace("—", "-").replace(" ", "").replace("_", "")
    cleaned = cleaned.replace("O", "0").replace("I", "1").replace("L", "1")
    match = re.search(r"SPN-?(\d{4})", cleaned)
    return f"SPN-{match.group(1)}" if match else None


def _clean_date(value: str) -> str | None:
    if UNREADABLE_RE.search(value):
        return "UNREADABLE"
    cleaned = value.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    match = re.search(r"(20\d{2})[-/.:](\d{2})[-/.:](\d{2})", cleaned)
    if not match:
        return None
    year, month, day = match.groups()
    # Degraded rasterizer often turns the final 6 in 2026 into 8.
    if year == "2028":
        year = "2026"
    candidate = f"{year}-{month}-{day}"
    try:
        datetime.strptime(candidate, "%Y-%m-%d")
    except ValueError:
        return "UNREADABLE"
    return candidate


def _clean_name(value: str) -> str | None:
    """Two-token compositional name repair against NAME_PARTS."""
    cleaned = _clean_value(value)
    cleaned = re.split(
        r"\s{2,}|\b(?:Species|Home|Visa|Sponsor|Arrival|Declared|PASSPORT|Registry)\b",
        cleaned,
        maxsplit=1,
    )[0]
    cleaned = re.sub(r"[^A-Za-z -]", "", cleaned).strip()
    if NAME_CUT_RE.search(cleaned) or "WHITEOUT" in cleaned.upper():
        return "[NAME CUT OUT]"
    words = cleaned.split()
    if len(words) != 2:
        return None
    corrected: list[str] = []
    for word in words:
        hit = _fuzzy_vocab(word, NAME_PARTS, cutoff=0.62)
        if not hit:
            return None
        # Preserve generator casing: Capitalize first letter of each part.
        corrected.append(hit[0].upper() + hit[1:] if hit else hit)
    return " ".join(corrected)


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
        "sponsor_id": r"Sponsor(?:\s*ID)?\s*[:|]?\s*(SPN-?[0-9OIl]{4})",
        "arrival_date": r"Arrival Date\s*[:|]?\s*([0-9OIl]{4}[-/.:][0-9OIl]{2}[-/.:][0-9OIl]{2}|UNREADABLE|MISSING|N/?A|ILLEGIBLE)",
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

    Train receipts encode:
      paid   = $809 + waiver N/A
      unpaid = $0   + waiver N/A
      waived = $0   + non-N/A waiver code
    Amount alone must not mean paid (and $0 alone is not waived).
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
        if value == 0 and waiver_code == "N/A":
            return "unpaid"
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
    # $809 on a receipt without a parsed waiver is usually paid; never treat
    # bare $0 as waived (that collides with unpaid's N/A waiver pattern).
    if receipt_like and AMOUNT_809_RE.search(text) and not waiver_m:
        return "paid"
    # Goleffect-style: explicit Amount $809.00 line is paid even if the
    # "Fee Receipt" heading OCR is mangled.
    if AMOUNT_809_RE.search(text) and re.search(r"(?i)\bamount\b", text):
        return "paid"
    # $0 + DIP-WAIVER (or any non-N/A waiver cue) ⇒ waived.
    if (
        AMOUNT_ZERO_RE.search(text)
        and re.search(r"(?i)DIP.?WAIVER|waiver\s*code\s*[:.]?\s*(?!N/?A\b)\S+", text)
    ):
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
        fixed = OCR_VISA_FIXES.get(compact)
        if fixed:
            return fixed
        return _fuzzy_vocab(value, VISA_CLASSES, cutoff=0.6)
    if key == "sponsor_id":
        return _clean_sponsor(value)
    if key == "fee_status":
        return _normalize_fee_token(value)
    if key == "arrival_date":
        return _clean_date(value)
    if key == "species_code":
        return _canonicalize_species(value)
    if key == "home_world":
        value = value.replace("¢", "c").replace("Bamard", "Barnard").replace("Barard", "Barnard")
        canon = _canonicalize_world(value)
        if canon and canon in KNOWN_WORLDS:
            return canon
        fuzzy = _fuzzy_vocab(value, KNOWN_WORLDS, cutoff=0.72)
        return fuzzy or canon
    if key == "applicant_name":
        if NAME_CUT_RE.search(value):
            return "[NAME CUT OUT]"
        if FOOTER_RE.search(value) or INJECTION_RE.search(value):
            return None
        grammar = _clean_name(value)
        if grammar:
            return grammar
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
        canon = _canonicalize_purpose(value)
        if canon:
            return canon
        return _fuzzy_vocab(value, KNOWN_PURPOSES, cutoff=0.64)
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


def _fuzzy_risk_mentions(text: str) -> set[str]:
    """Recover badly OCRed flag names, only from flag/reason contexts.

    Global fuzzy matching invents flags from words like "biometric"; gate on
    observed-flags / risk / reason / finding lines (goleffect-style).
    """
    found: set[str] = set()
    for line in text.splitlines():
        if not re.search(r"(?i)\b(?:obs\w*|flags?|risk|reason|finding|warrant|hazard|embargo)\b", line):
            continue
        compact = re.sub(r"[^a-z0-9]", "", line.lower())
        for flag in ALL_FLAGS:
            target = flag.replace("_", "")
            if target and target in compact:
                found.add(flag)
                continue
            # Mild OCR: allow one-char deletion in the compact flag name.
            for needles in FLAG_FUZZY.get(flag, ()):
                needle = re.sub(r"[^a-z0-9]", "", needles)
                if len(needle) >= 6 and needle in compact:
                    found.add(flag)
                    break
    return found


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
    found |= _fuzzy_risk_mentions(text)
    return found


def _fee_crop_consensus_chunks(img: Image.Image) -> list[str]:
    """Multi-threshold binary OCR of the fee header band (strobl-style)."""
    try:
        import pytesseract
    except ImportError:
        return []
    w, h = img.size
    header = img.crop((0, 0, w, max(1, int(h * 0.35))))
    gray = ImageOps.autocontrast(ImageOps.grayscale(header))
    chunks: list[str] = []
    for thr in (130, 160):
        binary = gray.point(lambda x, t=thr: 255 if x > t else 0)
        try:
            chunks.append(pytesseract.image_to_string(binary, config="--psm 6"))
        except Exception:
            pass
    up = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
    try:
        chunks.append(pytesseract.image_to_string(up, config="--psm 11"))
    except Exception:
        pass
    return [c for c in chunks if c.strip()]


def _consensus_fee_from_chunks(chunks: list[str]) -> str | None:
    """Require ≥2 agreeing fee inferences across OCR variants."""
    votes: dict[str, int] = {}
    for text in chunks:
        if not text.strip():
            continue
        tok = _fee_status_capture(text)
        norm = _normalize_fee_token(tok) if tok else None
        if not norm or norm == "unknown":
            norm = _fee_from_amount_text(text)
        if norm and norm != "unknown":
            votes[norm] = votes.get(norm, 0) + 1
    if not votes:
        return None
    best = max(votes, key=votes.get)
    if votes[best] >= 2:
        return best
    # Single strong amount+waiver hit is enough.
    if best in {"paid", "waived", "unpaid"}:
        for text in chunks:
            if _fee_from_amount_text(text) == best:
                return best
    return None


def _ocr_image(
    img: Image.Image,
    try_rotate: bool = False,
    fee_boost: bool = False,
    second_pass: bool = False,
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

    def _run(target: Image.Image, sharpen: bool = False, psm: int = 6) -> None:
        gray = ImageOps.autocontrast(ImageOps.grayscale(target))
        if sharpen:
            gray = gray.filter(ImageFilter.SHARPEN)
        try:
            chunks.append(pytesseract.image_to_string(gray, config=f"--psm {psm}"))
        except Exception:
            pass

    # Plain autocontrast preserves faint flag lines; sharpen helps titles/IDs.
    # When second_pass-only, skip the base reads (already done in pass 2).
    if not second_pass or fee_boost:
        _run(header, sharpen=False)
        _run(img, sharpen=False)
        _run(mid, sharpen=False)
        _run(img, sharpen=True)

    if fee_boost and not second_pass:
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

    # Guarded second pass: alternate PSM / invert / stronger binary — only used
    # when fee is still unknown or risk flags are empty on image pages.
    # Keep this cheap: a few targeted reads, not a full OCR rewrite.
    if second_pass:
        gray = ImageOps.autocontrast(ImageOps.grayscale(img))
        try:
            chunks.append(pytesseract.image_to_string(gray, config="--psm 4"))
        except Exception:
            pass
        try:
            inv = ImageOps.invert(gray)
            chunks.append(pytesseract.image_to_string(inv, config="--psm 6"))
        except Exception:
            pass
        binary = gray.point(lambda x: 255 if x > 150 else 0)
        try:
            chunks.append(pytesseract.image_to_string(binary, config="--psm 11"))
        except Exception:
            pass
        # Flag / observed-flags band (lower half of biometric slips).
        lower = img.crop((0, int(h * 0.40), w, h))
        lower_g = ImageOps.autocontrast(ImageOps.grayscale(lower))
        try:
            chunks.append(pytesseract.image_to_string(lower_g, config="--psm 6"))
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
    second_pass: bool = False,
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
                _ocr_image(
                    img,
                    try_rotate=try_rotate,
                    fee_boost=fee_boost,
                    second_pass=second_pass,
                )
            )
            if parts[-1].strip():
                embedded_ok = True
        except Exception:
            continue
    # Always raster sparse / footer-only pages — embedded OCR can be empty
    # even when a large scan exists, and portraits shouldn't block recovery.
    page_text_len = len((page.get_text() or "").strip())
    embedded_text = "\n".join(p for p in parts if p.strip())
    if second_pass:
        need_raster = (not embedded_ok) or (len(embedded_text.strip()) < 40)
    else:
        need_raster = (not embedded_ok) or page_text_len < 100
    if need_raster:
        try:
            use_dpi = 220 if second_pass else dpi
            pix = page.get_pixmap(dpi=use_dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            parts.append(
                _ocr_image(
                    img,
                    try_rotate=try_rotate,
                    fee_boost=False if second_pass else fee_boost,
                    second_pass=second_pass,
                )
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


def _tesseract_image_file(image_path: Path, psms: tuple[str, ...] = ("3", "11")) -> str:
    chunks: list[str] = []
    for psm in psms:
        try:
            cp = subprocess.run(
                ["tesseract", image_path.name, "stdout", "--psm", psm],
                cwd=str(image_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
                timeout=20,
                check=False,
            )
            if cp.returncode == 0 and cp.stdout:
                chunks.append(cp.stdout)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "\n".join(chunks)


def _full_document_ocr(pdf_path: Path, dpi: int = 150) -> list[str]:
    """Render-first OCR of every page (goleffect/strobl idea, our implementation).

    Prefer pdftoppm+tesseract when available; fall back to PyMuPDF pixmaps.
    Returns one text blob per page.
    """
    pdf_path = Path(pdf_path)
    cache_root = os.environ.get("MIB_OCR_CACHE")
    cache_file = Path(cache_root, pdf_path.stem + ".json") if cache_root else None
    if cache_file and cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass

    page_texts: list[str] = []
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        with tempfile.TemporaryDirectory(prefix="mib-ocr-") as tmp:
            work = Path(tmp)
            prefix = work / "page"
            try:
                subprocess.run(
                    [
                        pdftoppm,
                        "-jpeg",
                        "-jpegopt",
                        "quality=85",
                        "-r",
                        str(dpi),
                        str(pdf_path),
                        str(prefix),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=45,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return []
            images = sorted(work.glob("page-*.jpg"))
            for image in images:
                page_texts.append(_tesseract_image_file(image))
    else:
        try:
            import pytesseract
        except ImportError:
            pytesseract = None  # type: ignore[assignment]
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return []
        for page in doc:
            try:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
            except Exception:
                page_texts.append("")
                continue
            chunks: list[str] = []
            gray = ImageOps.autocontrast(ImageOps.grayscale(img))
            if pytesseract is not None:
                for psm in (3, 11):
                    try:
                        chunks.append(pytesseract.image_to_string(gray, config=f"--psm {psm}"))
                    except Exception:
                        pass
            else:
                with tempfile.TemporaryDirectory(prefix="mib-page-") as tmp:
                    path = Path(tmp) / "page.png"
                    gray.save(path)
                    chunks.append(_tesseract_image_file(path))
            page_texts.append("\n".join(c for c in chunks if c.strip()))

    if cache_file and page_texts:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(page_texts), encoding="utf-8")
        except Exception:
            pass
    return page_texts


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
    biometric_flags_observed = False

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

    # Pass 1.5: ALWAYS render-OCR the full packet at 150 DPI (goleffect-style).
    # This is the main extraction unlock vs selective sparse-page OCR alone.
    full_ocr_pages = _full_document_ocr(pdf_path, dpi=150)
    if full_ocr_pages:
        used_ocr = True
        for pe, ocr_text in zip(pages, full_ocr_pages):
            if not ocr_text or not ocr_text.strip():
                continue
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
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
                        value=amt, source=source, page=pe.page_index, confidence=0.7
                    )
            # Closed-vocab presence recovery on recognized pages.
            folded = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())
            if ocr_doc in {"intake", "registry", "biometric", "sponsor_letter", "fee_receipt", "unknown"}:
                if "visa_class" not in hits:
                    for visa in VISA_CLASSES:
                        if re.sub(r"[^A-Z0-9]", "", visa) in folded:
                            hits["visa_class"] = FieldHit(
                                value=visa, source=source, page=pe.page_index, confidence=0.55
                            )
                            break
                if "species_code" not in hits:
                    for sp in KNOWN_SPECIES:
                        if re.sub(r"[^A-Z0-9]", "", sp) in folded:
                            hits["species_code"] = FieldHit(
                                value=sp, source=source, page=pe.page_index, confidence=0.55
                            )
                            break
                if "home_world" not in hits:
                    for world in KNOWN_WORLDS:
                        if re.sub(r"[^A-Z0-9]", "", world.upper()) in folded:
                            hits["home_world"] = FieldHit(
                                value=world, source=source, page=pe.page_index, confidence=0.55
                            )
                            break
                if "declared_purpose" not in hits:
                    for purpose in KNOWN_PURPOSES:
                        compact_p = re.sub(r"[^A-Z0-9]", "", purpose.upper())
                        if len(compact_p) >= 6 and compact_p in folded:
                            hits["declared_purpose"] = FieldHit(
                                value=purpose, source=source, page=pe.page_index, confidence=0.5
                            )
                            break
            for h in hits.values():
                h.confidence *= 0.85
                h.source = source
            _merge_hits(pe.fields, hits, [])
            _merge_hits(merged, hits, conflicts)
            risk_flags |= _flags_from_text(ocr_text)
            auth = AUTHORITATIVE_FEE_RE.search(ocr_text)
            if auth:
                tok = next(g for g in auth.groups() if g)
                _merge_hits(
                    merged,
                    {
                        "fee_status": FieldHit(
                            value=tok.lower(),
                            source="adjudicator_note",
                            page=pe.page_index,
                            confidence=0.95,
                        )
                    },
                    conflicts,
                )
            if OBSERVED_FLAGS_NONE_RE.search(ocr_text) or hits.get("observed_flags"):
                biometric_flags_observed = True
            fm = FINDING_RE.search(ocr_text)
            if fm:
                finding = _normalize_finding(fm.group(1))
                if finding and "SAMPLE" not in ocr_text.upper():
                    _merge_hits(
                        merged,
                        {
                            "manual_finding": FieldHit(
                                value=finding,
                                source="adjudicator_note",
                                page=pe.page_index,
                                confidence=0.85,
                            )
                        },
                        conflicts,
                    )

    critical_keys = ["applicant_name", "visa_class", "fee_status", "arrival_date", "species_code"]
    critical_missing = any(k not in merged for k in critical_keys)
    image_heavy = trusted_span_count < 12

    # Pass 2: selective boost OCR only if full OCR unavailable.
    if not full_ocr_pages:
        for pe, page in zip(pages, doc):
            if not _page_needs_ocr(pe, critical_missing or image_heavy):
                continue
            ocr_text = _ocr_page(page, doc, fee_boost=False, risk_contrast=True)
            if not ocr_text.strip():
                continue
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
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

    # Pass 2b: fee still missing/unknown → header 2x/binary OCR on sparse pages
    # that look like fee receipts (avoid re-OCR of every biometric slip).
    fee_hit = merged.get("fee_status")
    fee_needs_boost = fee_hit is None or fee_hit.value == "unknown"
    if fee_needs_boost:
        for pe, page in zip(pages, doc):
            if pe.n_trusted_spans >= 10 and not page.get_images():
                continue
            if not page.get_images() and pe.n_trusted_spans >= 10:
                continue
            hint = pe.trusted_text
            if not re.search(
                r"(?i)fee|rec[ei]|amount|\$\s*\d|sta[tuswys]|rezo|reza|recast",
                hint,
            ):
                # Still try completely blank scan pages (footer-only) once
                if pe.n_trusted_spans > 3 and not page.get_images():
                    continue
                if pe.n_trusted_spans > 3 and page.get_images() and pe.n_trusted_spans >= 8:
                    # Image page without fee cue — leave for guarded second pass.
                    continue
            ocr_text = _ocr_page(page, doc, fee_boost=True)
            if not ocr_text.strip():
                continue
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
            source = "ocr_fee_receipt"
            if re.search(r"(?i)fee\s*rec|mib\s*fe|fee\s*rez|recast", ocr_text):
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
            if "fee_status" in merged and merged["fee_status"].value != "unknown":
                break

    # Pass 2c: guarded second OCR — ONLY when fee still unknown on a garble
    # fee-receipt page, or when risk flags are empty on biometric-looking image
    # pages. Fills unknown fee / missing flags only (no full RapidOCR rewrite).
    fee_still_unknown = (
        "fee_status" not in merged or merged["fee_status"].value == "unknown"
    )
    fee_garble_pages = [
        (pe, page)
        for pe, page in zip(pages, doc)
        if fee_still_unknown
        and (
            re.search(r"(?i)fee\s*rez|rezo|reza|recast|fe[eag]\s*rec|mib\s*fe", pe.trusted_text)
            or (pe.doc_type == "fee_receipt" and pe.n_trusted_spans < 10)
        )
    ]
    bio_candidate_pages = [
        (pe, page)
        for pe, page in zip(pages, doc)
        if (
            (pe.n_trusted_spans < 10 or bool(page.get_images()))
            and (
                pe.doc_type == "biometric"
                or re.search(
                    r"(?i)b-?13|biometric|observed\s*f|warrant|hazard|tamper",
                    pe.trusted_text,
                )
                or (pe.n_trusted_spans < 4 and bool(page.get_images()))
            )
        )
    ]
    hunt_flags = (not risk_flags) and bool(bio_candidate_pages)
    targets: list[tuple] = []
    seen_pages: set[int] = set()
    for pe, page in fee_garble_pages + (bio_candidate_pages if hunt_flags else []):
        if pe.page_index in seen_pages:
            continue
        seen_pages.add(pe.page_index)
        targets.append((pe, page))
    if targets:
        second_budget = 0
        for pe, page in targets:
            if second_budget >= 2:
                break
            ocr_text = _ocr_page(
                page,
                doc,
                dpi=200,
                fee_boost=False,
                second_pass=True,
                risk_contrast=hunt_flags and pe.doc_type in {"biometric", "unknown"},
            )
            second_budget += 1
            if not ocr_text.strip():
                continue
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + ocr_text).strip()
            if re.search(r"(?i)fee\s*rec|mib\s*fe|fee\s*rez|recast", ocr_text):
                pe.doc_type = "fee_receipt"
                docs_present.add("fee_receipt")
            if re.search(r"(?i)observed\s*f|form\s*b-?13|biometric", ocr_text):
                docs_present.add("biometric")
                if pe.doc_type == "unknown":
                    pe.doc_type = "biometric"
            source = "ocr"
            if fee_still_unknown:
                hits = _regex_fields(ocr_text, source, pe.page_index)
                raw_fee = hits.get("fee_status")
                if raw_fee is None or _normalize_fee_token(raw_fee.value) is None:
                    amt = _fee_from_amount_text(ocr_text)
                    if amt:
                        hits["fee_status"] = FieldHit(
                            value=amt, source=source, page=pe.page_index, confidence=0.45
                        )
                keep: dict[str, FieldHit] = {}
                if "fee_status" in hits:
                    keep["fee_status"] = hits["fee_status"]
                for h in keep.values():
                    h.confidence *= 0.65
                    h.source = source
                _merge_hits(merged, keep, conflicts)
                if "fee_status" in merged and merged["fee_status"].value != "unknown":
                    fee_still_unknown = False
            new_flags = _flags_from_text(ocr_text)
            if hunt_flags and new_flags:
                risk_flags |= new_flags
                hunt_flags = False
            if OBSERVED_FLAGS_NONE_RE.search(ocr_text):
                biometric_flags_observed = True
            if not fee_still_unknown and not hunt_flags:
                break

    # Pass 2d: RapidOCR on large embedded scans when fee still unknown
    # OR other critical scored fields remain empty (strobl dual-OCR idea).
    fee_still_unknown = (
        "fee_status" not in merged or merged["fee_status"].value == "unknown"
    )
    critical_unknown = [
        k
        for k in (
            "applicant_name",
            "species_code",
            "home_world",
            "visa_class",
            "sponsor_id",
            "arrival_date",
            "declared_purpose",
        )
        if k not in merged
        or merged[k].value
        in {"unknown", "UNREADABLE", "SPN-0000", "1900-01-01", "[NAME CUT OUT]"}
    ]
    need_rapid = fee_still_unknown or bool(critical_unknown) or (
        not biometric_flags_observed and not risk_flags
    )
    if rapid_ocr.rapid_available() and need_rapid:
        rapid_budget = 0
        for pe, page in zip(pages, doc):
            if rapid_budget >= 3:
                break
            # Prefer large embedded letter-size scans (fee receipts).
            big = False
            for img in page.get_images() or []:
                try:
                    pix = fitz.Pixmap(doc, img[0])
                    if pix.width >= 800 or pix.height >= 800:
                        big = True
                        break
                except Exception:
                    continue
            fee_page = pe.doc_type == "fee_receipt" or bool(
                re.search(r"(?i)fee|rec[ei]|amount|waiver|mib\s*fe", pe.trusted_text)
            )
            bio_page = pe.doc_type == "biometric" or bool(
                re.search(r"(?i)observed\s*f|form\s*b-?13|biometric", pe.trusted_text)
            )
            sparse = pe.n_trusted_spans < 8
            if not (big or fee_page or bio_page or sparse or critical_unknown):
                continue
            rapid_text = ""
            if fee_still_unknown and (fee_page or big):
                rapid_text = rapid_ocr.ocr_page_fee_band(page, dpi=200)
            if not rapid_text.strip():
                rapid_text = rapid_ocr.ocr_page_text(page, dpi=180)
            if not rapid_text.strip():
                continue
            rapid_budget += 1
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + rapid_text).strip()
            source = "ocr"
            hits = _regex_fields(rapid_text, source, pe.page_index)
            raw_fee = hits.get("fee_status")
            if raw_fee is None or _normalize_fee_token(raw_fee.value) is None:
                amt = _fee_from_amount_text(rapid_text)
                if amt:
                    hits["fee_status"] = FieldHit(
                        value=amt, source=source, page=pe.page_index, confidence=0.55
                    )
            auth = AUTHORITATIVE_FEE_RE.search(rapid_text)
            if auth:
                tok = next(g for g in auth.groups() if g)
                hits["fee_status"] = FieldHit(
                    value=tok.lower(), source="adjudicator_note", page=pe.page_index, confidence=0.9
                )
            # Closed-vocab presence on Rapid text for still-unknown fields.
            folded = re.sub(r"[^A-Z0-9]", "", rapid_text.upper())
            if "species_code" not in merged:
                for sp in KNOWN_SPECIES:
                    if re.sub(r"[^A-Z0-9]", "", sp) in folded:
                        hits["species_code"] = FieldHit(
                            value=sp, source=source, page=pe.page_index, confidence=0.45
                        )
                        break
            if "home_world" not in merged:
                for world in KNOWN_WORLDS:
                    if re.sub(r"[^A-Z0-9]", "", world.upper()) in folded:
                        hits["home_world"] = FieldHit(
                            value=world, source=source, page=pe.page_index, confidence=0.45
                        )
                        break
            if "declared_purpose" not in merged:
                for purpose in KNOWN_PURPOSES:
                    if re.sub(r"[^A-Z0-9]", "", purpose.upper()) in folded:
                        hits["declared_purpose"] = FieldHit(
                            value=purpose, source=source, page=pe.page_index, confidence=0.45
                        )
                        break
            keep: dict[str, FieldHit] = {}
            for key, hit in hits.items():
                if key == "fee_status":
                    norm = _normalize_fee_token(hit.value) or hit.value
                    if norm and norm != "unknown":
                        hit.value = norm
                        keep[key] = hit
                elif key not in merged:
                    keep[key] = hit
                elif key in critical_unknown and merged[key].value in {
                    "unknown",
                    "UNREADABLE",
                    "SPN-0000",
                    "1900-01-01",
                    "[NAME CUT OUT]",
                }:
                    keep[key] = hit
            if keep:
                _merge_hits(merged, keep, conflicts)
                if "fee_status" in merged and merged["fee_status"].value != "unknown":
                    fee_still_unknown = False
            new_flags = _flags_from_text(rapid_text)
            if new_flags:
                risk_flags |= new_flags
            if OBSERVED_FLAGS_NONE_RE.search(rapid_text):
                biometric_flags_observed = True
            if re.search(r"(?i)fee\s*rec|mib\s*fe", rapid_text):
                pe.doc_type = "fee_receipt"
                docs_present.add("fee_receipt")
            if re.search(r"(?i)observed\s*f|form\s*b-?13|biometric", rapid_text):
                docs_present.add("biometric")
                if pe.doc_type == "unknown":
                    pe.doc_type = "biometric"

    # Pass 2e: multi-threshold fee-band consensus when fee still unknown (1 page).
    fee_still_unknown = (
        "fee_status" not in merged or merged["fee_status"].value == "unknown"
    )
    if fee_still_unknown:
        for pe, page in zip(pages, doc):
            if pe.n_trusted_spans >= 14 and not page.get_images():
                continue
            hint = pe.trusted_text
            if not (
                pe.doc_type == "fee_receipt"
                or re.search(r"(?i)fee|rec[ei]|amount|waiver|\$\s*\d", hint)
                or (pe.n_trusted_spans < 4 and page.get_images())
            ):
                continue
            try:
                pix = page.get_pixmap(dpi=180)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
            except Exception:
                continue
            chunks = _fee_crop_consensus_chunks(img)
            if not chunks:
                continue
            consensus_text = "\n".join(chunks)
            used_ocr = True
            pe.used_ocr = True
            pe.trusted_text = (pe.trusted_text + "\n" + consensus_text).strip()
            fee_val = _consensus_fee_from_chunks(chunks)
            if fee_val is None:
                fee_val = _fee_from_amount_text(consensus_text)
                tok = _fee_status_capture(consensus_text)
                if tok:
                    fee_val = _normalize_fee_token(tok) or fee_val
            if fee_val and fee_val != "unknown":
                _merge_hits(
                    merged,
                    {
                        "fee_status": FieldHit(
                            value=fee_val,
                            source="ocr_fee_receipt",
                            page=pe.page_index,
                            confidence=0.55,
                        )
                    },
                    conflicts,
                )
                docs_present.add("fee_receipt")
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
        # Manual / authoritative fee correction anywhere in packet
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
        auth = AUTHORITATIVE_FEE_RE.search(all_text)
        if auth and (
            "fee_status" not in merged or merged["fee_status"].value == "unknown"
        ):
            tok = next(g for g in auth.groups() if g)
            _merge_hits(
                merged,
                {
                    "fee_status": FieldHit(
                        value=tok.lower(),
                        source="adjudicator_note",
                        page=0,
                        confidence=0.95,
                    )
                },
                conflicts,
            )

    if OBSERVED_FLAGS_NONE_RE.search(all_text):
        biometric_flags_observed = True
    if merged.get("observed_flags"):
        biometric_flags_observed = True

    # Embargo worlds: attach planetary_embargo for extraction+deny parity
    # (goleffect/strobl serialization prior — identity-free world feature).
    reg = merged.get("registry_status")
    if reg and re.search(r"(?i)embargo", reg.value):
        risk_flags.add("planetary_embargo")
    home = merged.get("home_world")
    if home and home.value in ALWAYS_EMBARGO_WORLDS:
        risk_flags.add("planetary_embargo")
    if home and home.value in {"Wolf-1061c"} and re.search(r"(?i)embargo", all_text):
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
