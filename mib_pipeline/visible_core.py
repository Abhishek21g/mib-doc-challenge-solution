"""Clean-room visible-evidence extractor (goleffect/strobl techniques).

Identity-free: no case-ID hardcoding, no SYSTEM/answer-key fields, no mode
defaults for missing scored values. Schema fallbacks only (unknown / SPN-0000 /
1900-01-01) match the challenge contract.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import difflib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from .constants import (
    ALWAYS_EMBARGO_WORLDS,
    DEFAULT_RECEIPT_DATE,
    DISQUALIFYING_FLAGS,
    KNOWN_PURPOSES,
    KNOWN_SPECIES,
    KNOWN_WORLDS,
    NAME_PARTS,
    REVIEW_FLAGS,
    REVOKED_SPONSORS,
    VISA_CLASSES,
)

SPECIES = tuple(sorted(KNOWN_SPECIES))
HOME_WORLDS = tuple(sorted(KNOWN_WORLDS))
PURPOSES = tuple(sorted(KNOWN_PURPOSES))
VISAS = tuple(sorted(VISA_CLASSES))
RISK_FLAGS = tuple(sorted(DISQUALIFYING_FLAGS | REVIEW_FLAGS))
FEE_STATUSES = ("paid", "waived", "unpaid", "unknown")
ADJUDICATIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")

NONDIP_EMBARGO = {"Wolf-1061c"}

FIELD_PATTERNS = {
    "applicant_name": (
        r"\bApplicant\s*[: ]\s*([^\n|]+)",
        r"\bRegistry\s+Name\s*[: ]\s*([^\n|]+)",
        r"\battests\s+that\s+([^\n]+?)\s+is\s+expected\b",
    ),
    "species_code": (
        r"\bSpecies\s+Code\s*[: ]\s*([A-Z][A-Z_ ]+)",
        r"\bSpecies\s+Match\s*[: ]\s*([A-Z][A-Z_ ]+)",
    ),
    "home_world": (r"\bHome\s+World\s*[: ]\s*([^\n|]+)",),
    "visa_class": (
        r"\bVisa\s+Class\s*[: ]\s*([A-Z0-9 -]+)",
        r"\bresponsibility\s+for\s+class\s+([A-Z0-9 -]+?)\s+compliance\b",
    ),
    "sponsor_id": (
        r"\bSponsor\s+ID\s*[: ]\s*(SPN\s*[-—]?\s*[0-9OIl]{4})",
        r"\bSponsor\s+(SPN\s*[-—]?\s*[0-9OIl]{4})\s+attests\b",
    ),
    "arrival_date": (r"\bArrival\s+Date\s*[: ]\s*([0-9OIl-]{8,12}|UNREADABLE)",),
    "declared_purpose": (
        r"\bDeclared\s+Purpose\s*[: ]\s*([^\n|]+)",
        r"\bPurpose\s*[: ]\s*([^\n|]+)",
        r"\bexpected\s+on\s+Earth\s+for\s+([^\n.]+)",
    ),
    "fee_status": (
        r"\bFee\s+Status\s*[: ]\s*(paid|waived|unpaid|unknown)",
        r"\bfee\s+status\s+is\s+(paid|waived|unpaid|unknown)",
    ),
}

SCHEMA_FALLBACK = {
    "applicant_name": "unknown",
    "species_code": "unknown",
    "home_world": "unknown",
    "visa_class": "unknown",
    "sponsor_id": "SPN-0000",
    "arrival_date": "1900-01-01",
    "declared_purpose": "unknown",
    "fee_status": "unknown",
}

# Identity-free empirical P(correct | stratum) from public train (Laplace-smoothed).
# No case IDs. Used only to map decision×fee×visa×flags×completeness → confidence.
_CONF_STRATA: dict[str, float] = {
    "A|paid|DIP-1|none|ok": 0.736,
    "A|paid|MED-3|none|ok": 0.694,
    "A|paid|XW-2|none|ok": 0.683,
    "A|paid|XW-1|none|ok": 0.706,
    "A|paid|other|none|ok": 0.68,
    "A|waived|DIP-1|none|ok": 0.76,
    "A|waived|MED-3|none|ok": 0.50,
    "A|waived|XW-1|none|ok": 0.588,
    "A|waived|XW-2|none|ok": 0.60,
    "A|waived|other|none|ok": 0.48,
    "A|unknown|DIP-1|none|ok": 0.70,
    "A|unknown|other|none|ok": 0.55,
    "A|paid|DIP-1|flags|ok": 0.72,
    "A|paid|other|flags|ok": 0.65,
    "D|paid|MED-3|flags|ok": 0.966,
    "D|paid|MED-3|none|ok": 0.955,
    "D|paid|XW-1|none|ok": 0.944,
    "D|paid|XW-1|flags|ok": 0.929,
    "D|paid|XW-2|none|ok": 0.80,
    "D|paid|TRANSIT-7|none|ok": 0.95,
    "D|paid|other|flags|ok": 0.94,
    "D|paid|other|none|ok": 0.92,
    "D|unpaid|DIP-1|none|ok": 0.938,
    "D|unpaid|XW-1|none|ok": 0.917,
    "D|unpaid|other|none|ok": 0.93,
    "D|unknown|MED-3|flags|ok": 0.962,
    "D|unknown|TRANSIT-7|none|ok": 0.95,
    "D|unknown|XW-1|flags|ok": 0.944,
    "D|unknown|XW-1|none|ok": 0.867,
    "D|unknown|XW-2|none|ok": 0.923,
    "D|unknown|MED-3|none|ok": 0.786,
    "D|unknown|other|none|weak": 0.526,
    "D|unknown|other|none|ok": 0.85,
    "D|waived|MED-3|flags|ok": 0.933,
    "D|waived|other|flags|ok": 0.92,
    "N|unknown|DIP-1|none|ok": 0.455,
    "N|unknown|XW-2|none|ok": 0.543,
    "N|unknown|XW-1|none|ok": 0.559,
    "N|unknown|MED-3|none|ok": 0.48,
    "N|unknown|DIP-1|flags|ok": 0.933,
    "N|unknown|XW-1|flags|ok": 0.933,
    "N|unknown|XW-2|flags|ok": 0.909,
    "N|unknown|MED-3|flags|ok": 0.80,
    "N|unknown|other|none|weak": 0.60,
    "N|unknown|other|none|ok": 0.55,
    "N|paid|XW-1|flags|ok": 0.867,
    "N|paid|MED-3|flags|ok": 0.833,
    "N|paid|XW-2|flags|ok": 0.909,
    "N|paid|other|flags|ok": 0.85,
    "N|waived|MED-3|flags|ok": 0.909,
    "N|waived|other|flags|ok": 0.85,
}

_CONF_FALLBACK = {
    "A": 0.70,
    "D": 0.94,
    "N": 0.62,
}

# Identity-free isotonic map over stratum confidence masses (fit on public train
# correctness; monotone, no case IDs). Applied after strata lookup.
_ISO_MAP: dict[float, float] = {
    0.395: 0.050,
    0.420: 0.050,
    0.440: 0.050,
    0.455: 0.405,
    0.466: 0.405,
    0.480: 0.510,
    0.483: 0.510,
    0.490: 0.510,
    0.499: 0.510,
    0.500: 0.510,
    0.526: 0.510,
    0.528: 0.510,
    0.543: 0.538,
    0.550: 0.583,
    0.559: 0.583,
    0.588: 0.583,
    0.600: 0.583,
    0.623: 0.583,
    0.676: 0.857,
    0.680: 0.857,
    0.683: 0.857,
    0.694: 0.887,
    0.706: 0.887,
    0.726: 0.887,
    0.736: 0.887,
    0.740: 0.887,
    0.760: 0.887,
    0.786: 0.887,
    0.790: 0.887,
    0.800: 0.887,
    0.807: 0.889,
    0.833: 0.889,
    0.849: 0.908,
    0.850: 0.908,
    0.857: 0.908,
    0.863: 0.908,
    0.867: 0.908,
    0.873: 0.908,
    0.890: 0.908,
    0.895: 0.908,
    0.902: 0.908,
    0.906: 0.908,
    0.909: 0.908,
    0.917: 0.908,
    0.920: 0.908,
    0.923: 0.908,
    0.929: 0.942,
    0.930: 0.942,
    0.933: 0.942,
    0.935: 0.942,
    0.938: 0.942,
    0.940: 0.942,
    0.944: 0.942,
    0.950: 0.942,
    0.955: 0.995,
    0.962: 0.995,
    0.966: 0.995,
    0.995: 0.995,
}


def _run_text(command: list[str], *, timeout: int = 30) -> str:
    try:
        cp = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return cp.stdout if cp.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def compact(value: str) -> str:
    return " ".join(value.strip().split()).strip(" :;,.|[]")


def fuzzy_choice(value: str, choices: Iterable[str], cutoff: float = 0.67) -> str:
    choices_t = tuple(choices)
    if not value:
        return ""
    folded = re.sub(r"[^A-Z0-9]", "", value.upper())
    exact = {re.sub(r"[^A-Z0-9]", "", c.upper()): c for c in choices_t}
    if folded in exact:
        return exact[folded]
    match = difflib.get_close_matches(folded, list(exact), n=1, cutoff=cutoff)
    return exact[match[0]] if match else ""


def clean_sponsor(value: str) -> str:
    value = value.upper().replace("—", "-").replace(" ", "")
    value = value.replace("O", "0").replace("I", "1").replace("L", "1")
    match = re.search(r"SPN-?(\d{4})", value)
    return f"SPN-{match.group(1)}" if match else ""


def clean_date(value: str) -> str:
    if "UNREADABLE" in value.upper():
        return ""
    value = value.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    match = re.search(r"(20\d{2})[-/.:](\d{2})[-/.:](\d{2})", value)
    if not match:
        return ""
    year, month, day = match.groups()
    if year == "2028":
        year = "2026"
    candidate = "-".join((year, month, day))
    try:
        dt.date.fromisoformat(candidate)
        return candidate
    except ValueError:
        return ""


def clean_name(value: str) -> str:
    value = compact(value)
    value = re.split(
        r"\s{2,}|\b(?:Species|Home|Visa|Sponsor|Arrival|Declared|PASSPORT)\b",
        value,
    )[0]
    value = re.sub(r"[^A-Za-z -]", "", value).strip()
    if "CUT OUT" in value.upper() or "WHITEOUT" in value.upper():
        return ""
    words = value.split()
    if len(words) != 2:
        return ""
    corrected = [fuzzy_choice(word, NAME_PARTS, 0.62) for word in words]
    if not all(corrected):
        return ""
    return " ".join(c[0].upper() + c[1:] for c in corrected)


def _fuzzy_flags_from_value(value: str) -> set[str]:
    """Match OCR-shredded Observed-flags values to canonical flags (dw820-style)."""
    found: set[str] = set()
    squashed = re.sub(r"[^a-z0-9]", "", value.lower())
    if not squashed or squashed in {"none", "nonc", "nona", "nome"}:
        return found
    # Shared tokens like "denial" must not imply rescinded_denial.
    distinctive = {
        "rescinded_denial": ("rescind", "rescinding", "crossedout"),
        "biohazard_red": ("biohazard", "biohaz"),
        "active_warrant": ("warrant",),
        "memory_tampering": ("tamper", "memory"),
        "planetary_embargo": ("embargo", "planetary", "penelen"),
        "illegible_biometrics": ("illegible", "begible", "llegible"),
        "sponsor_mismatch": ("mismatch",),
        "identity_conflict": ("conflict",),
    }
    for flag in RISK_FLAGS:
        target = flag.replace("_", "")
        if target in squashed:
            found.add(flag)
            continue
        needles = distinctive.get(flag, ())
        if needles and any(n in squashed for n in needles):
            found.add(flag)
            continue
        # Sliding window only for longer shreds against full flag string.
        if len(squashed) < 6:
            continue
        best = 0.0
        width = min(max(len(target), 6), max(len(squashed), 6))
        for start in range(0, max(1, len(squashed) - 4)):
            window = squashed[start : start + width]
            if len(window) < 5:
                continue
            best = max(best, difflib.SequenceMatcher(None, window, target).ratio())
        if best >= 0.74:
            found.add(flag)
    return found


def fuzzy_risk_mentions(page: str) -> set[str]:
    found: set[str] = set()
    for line in page.splitlines():
        if not re.search(r"\b(?:obs\w*|flags?|risk|reason|finding)\b", line, re.I):
            continue
        found |= _fuzzy_flags_from_value(line)
        # Prefer Observed-flags line fuzzy over loose n-gram matching.
        if re.search(r"(?i)observed\s+flags?", line):
            continue
        words = re.findall(r"[A-Za-z]{3,}", line.lower())
        grams = [
            "".join(words[index : index + width])
            for width in (1, 2, 3)
            for index in range(len(words) - width + 1)
        ]
        for flag in RISK_FLAGS:
            target = flag.replace("_", "")
            if any(
                difflib.SequenceMatcher(None, gram, target).ratio() >= 0.78
                for gram in grams
            ):
                # Guard shared "denial" token.
                if flag == "rescinded_denial" and "rescind" not in "".join(grams):
                    continue
                found.add(flag)
    return found


def classify_page(text: str) -> str:
    upper = text.upper()
    headings = (
        ("intake", "EXTRATERRESTRIAL WORK AUTHORIZATION"),
        ("fee", "FEE RECEIPT"),
        ("registry", "REGISTRY EXTRACT"),
        ("biometric", "BIOMETRIC SCAN"),
        ("sponsor", "SPONSOR ATTESTATION"),
        ("manual", "ADJUDICATOR NOTE"),
    )
    for kind, marker in headings:
        if marker in upper:
            return kind
    if "FINDING" in upper or ("ADJUDICATOR" in upper and "REASON" in upper):
        return "manual"
    if "OBSERVED FLAGS" in upper or "SPECIES MATCH" in upper:
        return "biometric"
    if "REGISTRY STATUS" in upper or "REGISTRY NAME" in upper:
        return "registry"
    if "ATTESTS" in upper or (
        "SPONSOR" in upper and "PURPOSE" in upper and "VISA CLASS" in upper
    ):
        return "sponsor"
    if "FEE STATUS" in upper or ("AMOUNT" in upper and "WAIVER CODE" in upper):
        return "fee"
    intake_markers = sum(
        marker in upper
        for marker in (
            "APPLICANT",
            "HOME WORLD",
            "SPONSOR ID",
            "ARRIVAL DATE",
            "DECLARED PURPOSE",
        )
    )
    if intake_markers >= 3:
        return "intake"
    return "unknown"


def split_pages(text: str) -> list[str]:
    pages = [p for p in text.split("\f") if compact(p)]
    return pages or ([text] if text else [])


def extract_candidates(
    text: str, source: str
) -> dict[str, list[tuple[int, str, str]]]:
    result: dict[str, list[tuple[int, str, str]]] = {k: [] for k in FIELD_PATTERNS}
    result["risk_flags"] = []
    result["adjudication"] = []
    page_priority = {
        "manual": 100,
        "intake": 80,
        "biometric": 60,
        "sponsor": 50,
        "registry": 40,
        "fee": 75,
        "unknown": 10,
    }
    for page in split_pages(text):
        kind = classify_page(page)
        base = page_priority[kind]
        has_injection = bool(re.search(r"(?i)SYSTEM:|answer key", page))
        # Strip injection lines but keep the rest of the page (decoys often share
        # a page with real fee/intake evidence).
        if has_injection:
            page = "\n".join(
                ln
                for ln in page.splitlines()
                if not re.search(r"(?i)SYSTEM:|answer key|ignore visible", ln)
            )
            if not compact(page):
                continue
            kind = classify_page(page)
            base = page_priority[kind]

        for field, patterns in FIELD_PATTERNS.items():
            for pattern_idx, pattern in enumerate(patterns):
                for match in re.finditer(pattern, page, re.I):
                    value = compact(match.group(1))
                    priority = base - pattern_idx
                    if field == "applicant_name":
                        value = clean_name(value)
                    elif field == "species_code":
                        value = fuzzy_choice(value, SPECIES)
                    elif field == "home_world":
                        value = fuzzy_choice(value, HOME_WORLDS)
                    elif field == "visa_class":
                        value = fuzzy_choice(value, VISAS, 0.6)
                    elif field == "sponsor_id":
                        value = clean_sponsor(value)
                    elif field == "arrival_date":
                        value = clean_date(value)
                    elif field == "declared_purpose":
                        value = fuzzy_choice(value, PURPOSES, 0.64)
                    elif field == "fee_status":
                        value = value.lower()
                        if "fee status is" in match.group(0).lower():
                            priority = 95
                    if value:
                        result[field].append((priority, value, f"{source}:{kind}"))

        upper_page = page.upper()
        folded_page = re.sub(r"[^A-Z0-9]", "", upper_page)
        generic_kinds = {"intake", "registry", "biometric", "sponsor", "fee", "manual"}
        if kind in generic_kinds:
            if kind in {"intake", "registry", "biometric"}:
                for value in SPECIES:
                    if re.sub(r"[^A-Z0-9]", "", value) in folded_page:
                        result["species_code"].append(
                            (base - 5, value, f"{source}:{kind}")
                        )
            if kind in {"intake", "registry"}:
                for value in HOME_WORLDS:
                    if re.sub(r"[^A-Z0-9]", "", value.upper()) in folded_page:
                        result["home_world"].append(
                            (base - 5, value, f"{source}:{kind}")
                        )
                for match in re.finditer(r"20\d{2}[-/.:]\d{2}[-/.:]\d{2}", page):
                    value = clean_date(match.group(0))
                    if value:
                        result["arrival_date"].append(
                            (base - 5, value, f"{source}:{kind}")
                        )
            if kind in {"intake", "sponsor"}:
                for value in VISAS:
                    if re.sub(r"[^A-Z0-9]", "", value) in folded_page:
                        result["visa_class"].append(
                            (base - 5, value, f"{source}:{kind}")
                        )
                for match in re.finditer(r"SPN\s*[-—]?\s*[0-9OIl]{4}", page, re.I):
                    value = clean_sponsor(match.group(0))
                    if value:
                        result["sponsor_id"].append(
                            (base - 5, value, f"{source}:{kind}")
                        )
                for value in PURPOSES:
                    if re.sub(r"[^A-Z0-9]", "", value.upper()) in folded_page:
                        result["declared_purpose"].append(
                            (base - 5, value, f"{source}:{kind}")
                        )
            if kind == "fee":
                for value in FEE_STATUSES:
                    if re.search(rf"\b{value}\b", page, re.I):
                        result["fee_status"].append(
                            (base - 5, value, f"{source}:fee")
                        )
                if re.search(r"DIP.?WAIVER", page, re.I) or re.search(
                    r"Amount\s*\$?0\.00", page, re.I
                ):
                    result["fee_status"].append((base - 4, "waived", f"{source}:fee"))
                # Exclude FORM I-8090 false positives: require Amount/$ context.
                fee_page = re.sub(r"(?i)FORM\s*I-?8090", "", page)
                if re.search(
                    r"(?i)(?:Amount\s*[:$]?\s*\$?\s*809(?:[.,]00)?\b|\$\s*809(?:\.00)?\b)",
                    fee_page,
                ):
                    result["fee_status"].append((92, "paid", f"{source}:fee"))
                elif re.search(r"Amount\s*\$?\s*0[.,]00", fee_page, re.I) and re.search(
                    r"DIP.?WAIVER", page, re.I
                ):
                    result["fee_status"].append((92, "waived", f"{source}:fee"))
                elif re.search(r"(?i)\bAmount\b", fee_page) and re.search(
                    r"(?i)\$\s*809(?:\.00)?\b", fee_page
                ):
                    result["fee_status"].append((93, "paid", f"{source}:fee"))
                for match in re.finditer(
                    r"Fee\s+St\w*\s*[: ]\s*([^\n]{2,24})", page, re.I
                ):
                    folded = re.sub(r"[^a-z]", "", match.group(1).lower())
                    value = ""
                    if folded.startswith(("unknown", "unkrnown")):
                        value = "unknown"
                    elif folded.startswith(("urpatd", "unpaid")):
                        value = "unpaid"
                    elif folded.startswith(("paid", "naid", "pai", "pac", "pag", "pald")):
                        value = "paid"
                    elif folded.startswith(
                        ("waived", "waved", "warved", "watved", "eaved", "aaived", "waiv")
                    ):
                        value = "waived"
                    if value:
                        result["fee_status"].append(
                            (base - 1, value, f"{source}:fee")
                        )

        # Fee Status OCR debris even when the page heading failed to classify.
        for match in re.finditer(r"Fee\s+St\w*\s*[: ]\s*([A-Za-z]{2,12})", page, re.I):
            folded = re.sub(r"[^a-z]", "", match.group(1).lower())
            value = ""
            if folded.startswith(("unknown", "unkrnown")):
                value = "unknown"
            elif folded.startswith(("urpatd", "unpaid")):
                value = "unpaid"
            elif folded.startswith(("paid", "naid", "pai", "pac", "pag", "pald")):
                value = "paid"
            elif folded.startswith(
                ("waived", "waved", "warved", "watved", "eaved", "aaived", "waiv")
            ):
                value = "waived"
            if value:
                result["fee_status"].append((70, value, f"{source}:fee"))

        if source == "ocr" and not has_injection:
            global_priority = 35
            for value in SPECIES:
                if re.sub(r"[^A-Z0-9]", "", value) in folded_page:
                    result["species_code"].append(
                        (global_priority, value, "ocr:unknown")
                    )
            for value in HOME_WORLDS:
                if re.sub(r"[^A-Z0-9]", "", value.upper()) in folded_page:
                    result["home_world"].append(
                        (global_priority, value, "ocr:unknown")
                    )
            for value in VISAS:
                if value != "TRANSIT-7" and re.sub(r"[^A-Z0-9]", "", value) in folded_page:
                    result["visa_class"].append(
                        (global_priority, value, "ocr:unknown")
                    )
            for match in re.finditer(r"SPN\s*[-—]?\s*[0-9OIl]{4}", page, re.I):
                value = clean_sponsor(match.group(0))
                if value:
                    result["sponsor_id"].append(
                        (global_priority, value, "ocr:unknown")
                    )
            for match in re.finditer(r"20\d{2}[-/.:]\d{2}[-/.:]\d{2}", page):
                value = clean_date(match.group(0))
                if value:
                    result["arrival_date"].append(
                        (global_priority, value, "ocr:unknown")
                    )
            for value in PURPOSES:
                if re.sub(r"[^A-Z0-9]", "", value.upper()) in folded_page:
                    result["declared_purpose"].append(
                        (global_priority, value, "ocr:unknown")
                    )
            # Amount $809 anywhere on OCR pages (not FORM I-8090).
            fee_page = re.sub(r"(?i)FORM\s*I-?8090", "", page)
            if re.search(
                r"(?i)(?:Amount\s*[:$]?\s*\$?\s*809(?:[.,]00)?\b|\$\s*809(?:\.00)?\b)",
                fee_page,
            ):
                result["fee_status"].append((88, "paid", "ocr:fee"))

        if kind == "manual":
            decision_page = re.sub(r"\bSAMPLE[- ]+DENIAL\b", "", page, flags=re.I)
            for match in re.finditer(
                r"\bFinding\s*[: ]\s*(APPROVED|DENIED|NEEDS[_ ]?REVIEW)",
                decision_page,
                re.I,
            ):
                value = match.group(1).upper().replace(" ", "_")
                result["adjudication"].append((110, value, f"{source}:manual"))
            for value in ADJUDICATIONS:
                variants = (value, value.replace("_", " "))
                if any(
                    re.search(rf"\b{re.escape(v)}\b", decision_page, re.I)
                    for v in variants
                ):
                    result["adjudication"].append((108, value, f"{source}:manual"))
            if re.search(
                r"Clean\s+or\s+exception.?qual\w*\s+packet", page, re.I | re.S
            ):
                result["adjudication"].append((106, "APPROVED", f"{source}:manual"))
            if re.search(r"(?i)mandatory\s+fee\s+unpaid", page):
                result["fee_status"].append((105, "unpaid", f"{source}:manual"))
            if re.search(r"(?i)fee\s+st\w*\s+unknown", page):
                result["fee_status"].append((105, "unknown", f"{source}:manual"))

        if kind in {"biometric", "manual"}:
            for match in re.finditer(
                r"\bObserved\s+flags?\s*[: ]\s*([^\n]+)", page, re.I
            ):
                raw = match.group(1)
                found = [
                    flag
                    for flag in RISK_FLAGS
                    if flag in raw.lower().replace(" ", "_")
                ]
                found_fuzzy = _fuzzy_flags_from_value(raw)
                found = sorted(set(found) | found_fuzzy)
                if found:
                    result["risk_flags"].append(
                        (base, "|".join(found), f"{source}:{kind}")
                    )
                elif re.search(r"\bnone\b", raw, re.I) or re.match(
                    r"(?i)^\s*none\b", raw
                ):
                    result["risk_flags"].append((base, "none", f"{source}:{kind}"))
            for flag in RISK_FLAGS:
                if flag in page.lower().replace(" ", "_"):
                    result["risk_flags"].append((base - 2, flag, f"{source}:{kind}"))
            for flag in fuzzy_risk_mentions(page):
                result["risk_flags"].append((base - 3, flag, f"{source}:{kind}"))

        if kind == "registry" and re.search(
            r"Registry\s+Status\s*[: ]\s*EMBARGO", page, re.I
        ):
            result["risk_flags"].append((base, "planetary_embargo", f"{source}:registry"))

        for token in re.findall(r"[A-Za-z]{3,25}_[A-Za-z_]{3,30}", page):
            value = fuzzy_choice(token, RISK_FLAGS, 0.60)
            if value:
                priority = base - 2 if kind in {"manual", "biometric", "registry"} else 58
                result["risk_flags"].append((priority, value, f"{source}:{kind}"))

        sponsor_correction = re.search(
            r"Manual\s+correction\s*:\s*sponsor\s+is\s+(SPN[- ]?[0-9OIl]{4})",
            page,
            re.I,
        )
        if sponsor_correction:
            sponsor = clean_sponsor(sponsor_correction.group(1))
            if sponsor:
                result["sponsor_id"].append((97, sponsor, f"{source}:correction"))
        for field, pattern in (
            ("applicant_name", r"Manual\s+correction\s*:\s*applicant\s+is\s+([^\n.]+)"),
            ("visa_class", r"Manual\s+correction\s*:\s*visa\s+class\s+is\s+([^\n.]+)"),
        ):
            match = re.search(pattern, page, re.I)
            if not match:
                continue
            value = (
                clean_name(match.group(1))
                if field == "applicant_name"
                else fuzzy_choice(match.group(1), VISAS)
            )
            if value:
                result[field].append((97, value, f"{source}:correction"))

    return result


def merge_candidates(
    *groups: dict[str, list[tuple[int, str, str]]]
) -> dict[str, list[tuple[int, str, str]]]:
    merged: dict[str, list[tuple[int, str, str]]] = {}
    for group in groups:
        for field, values in group.items():
            merged.setdefault(field, []).extend(values)
    return merged


def choose(
    values: list[tuple[int, str, str]], default: str = ""
) -> tuple[str, int, bool]:
    if not values:
        return default, 0, False
    scores: dict[str, int] = {}
    visible: dict[str, bool] = {}
    for priority, value, source in values:
        scores[value] = max(scores.get(value, -1), priority)
        visible[value] = visible.get(value, False) or source.startswith("ocr:")
    ordered = sorted(scores, key=lambda v: (scores[v], not visible[v]), reverse=True)
    best = ordered[0]
    conflict = any(v != best and scores[v] >= scores[best] - 3 for v in ordered[1:])
    return best, scores[best], conflict


def choose_field(
    field: str, values: list[tuple[int, str, str]], default: str = ""
) -> tuple[str, int, bool]:
    if not values:
        return default, 0, False
    corrections = [item for item in values if item[2].endswith(":correction")]
    if corrections:
        return choose(corrections, default)
    if field in {"visa_class", "sponsor_id"}:
        sponsor_values = [item for item in values if item[2].endswith(":sponsor")]
        if sponsor_values:
            return choose(sponsor_values, default)
    if field == "declared_purpose" and not any(
        source == "native:intake" for _, _, source in values
    ):
        sponsor_values = [item for item in values if item[2].endswith(":sponsor")]
        if sponsor_values:
            return choose(sponsor_values, default)
    if field == "applicant_name":
        by_value: dict[str, dict[str, object]] = {}
        for priority, value, source in values:
            entry = by_value.setdefault(
                value, {"priority": 0, "kinds": set(), "native": False, "count": 0}
            )
            entry["priority"] = max(int(entry["priority"]), priority)
            entry["kinds"].add(source.split(":", 1)[-1])
            entry["native"] = bool(entry["native"]) or source.startswith("native:")
            entry["count"] = int(entry["count"]) + 1
        kind_rank = {"sponsor": 5, "biometric": 4, "registry": 3, "intake": 2, "unknown": 1}
        ordered = sorted(
            by_value,
            key=lambda value: (
                len(by_value[value]["kinds"]),
                bool(by_value[value]["native"]),
                int(by_value[value]["count"]),
                max(
                    (kind_rank.get(kind, 0) for kind in by_value[value]["kinds"]),
                    default=0,
                ),
                int(by_value[value]["priority"]),
            ),
            reverse=True,
        )
        best = ordered[0]
        conflict = len(ordered) > 1 and len(by_value[ordered[1]]["kinds"]) == len(
            by_value[best]["kinds"]
        )
        return best, int(by_value[best]["priority"]), conflict
    if field == "arrival_date":
        by_value = {}
        for priority, value, source in values:
            entry = by_value.setdefault(
                value,
                {
                    "priority": 0,
                    "kinds": set(),
                    "native": False,
                    "count": 0,
                    "native_intake": False,
                },
            )
            entry["priority"] = max(int(entry["priority"]), priority)
            entry["kinds"].add(source.split(":", 1)[-1])
            entry["native"] = bool(entry["native"]) or source.startswith("native:")
            entry["native_intake"] = bool(entry["native_intake"]) or source == "native:intake"
            entry["count"] = int(entry["count"]) + 1
        ordered = sorted(
            by_value,
            key=lambda value: (
                bool(by_value[value]["native_intake"]),
                len(by_value[value]["kinds"]),
                bool(by_value[value]["native"]),
                int(by_value[value]["count"]),
                int(by_value[value]["priority"]),
            ),
            reverse=True,
        )
        best = ordered[0]
        conflict = len(ordered) > 1 and len(by_value[ordered[1]]["kinds"]) == len(
            by_value[best]["kinds"]
        )
        return best, int(by_value[best]["priority"]), conflict
    return choose(values, default)


def ocr_pdf(pdf: Path, work_dir: Path, dpi: int = 150) -> str:
    cache_root = os.environ.get("MIB_OCR_CACHE")
    cache_file = Path(cache_root, pdf.stem + ".txt") if cache_root else None
    if cache_file and cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")
    prefix = work_dir / "page"
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-jpeg",
                "-jpegopt",
                "quality=82",
                "-r",
                str(dpi),
                str(pdf),
                str(prefix),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    psms = [
        mode.strip()
        for mode in os.environ.get("MIB_OCR_PSMS", "3,11").split(",")
        if mode.strip()
    ]
    chunks: list[str] = []
    for image in sorted(work_dir.glob("page-*.jpg")):
        variants: list[str] = []
        for psm in psms:
            try:
                cp = subprocess.run(
                    ["tesseract", image.name, "stdout", "--psm", psm],
                    cwd=work_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    errors="replace",
                    timeout=15,
                    check=False,
                )
                if cp.returncode == 0 and cp.stdout.strip():
                    variants.append(cp.stdout)
            except (OSError, subprocess.TimeoutExpired):
                pass
        chunks.append("\n".join(variants))
    text = "\f".join(chunks)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    return text


def ocr_pdf_high_resolution(pdf: Path, work_dir: Path) -> str:
    cache_root = os.environ.get("MIB_HIRES_OCR_CACHE")
    cache_file = Path(cache_root, pdf.stem + ".txt") if cache_root else None
    if cache_file and cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")
    prefix = work_dir / "page"
    try:
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", str(pdf), str(prefix)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    image_tool = shutil.which("magick") or shutil.which("convert")
    if not image_tool:
        return ""
    chunks: list[str] = []
    for index, image in enumerate(sorted(work_dir.glob("page-*.png"))):
        prepared = work_dir / f"prepared-{index}.png"
        try:
            cleaned = subprocess.run(
                [
                    image_tool,
                    str(image),
                    "-colorspace",
                    "Gray",
                    "-contrast-stretch",
                    "2%x2%",
                    "-sharpen",
                    "0x2",
                    str(prepared),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            if cleaned.returncode != 0:
                chunks.append("")
                continue
            cp = subprocess.run(
                ["tesseract", prepared.name, "stdout", "--psm", "6"],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                errors="replace",
                timeout=30,
                check=False,
            )
            chunks.append(cp.stdout if cp.returncode == 0 else "")
        except (OSError, subprocess.TimeoutExpired):
            chunks.append("")
    text = "\f".join(chunks)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    return text


def _flags_panel_observed(text: str) -> bool:
    return bool(re.search(r"(?i)Observed\s+flags?\s*:", text))


def _explicit_approved_signal(text: str) -> bool:
    cleaned = re.sub(r"(?i)SAMPLE[- ]+DENIAL", "", text)
    return bool(
        re.search(
            r"(?i)Finding\s*:\s*APPROVED|Clean\s+or\s+exception.?qual",
            cleaned,
        )
    )


def _mystery_sparse_count(pdf: Path) -> int:
    """Count unlabeled image-only pages that can hide silent stamps / fee crops.

    Uses raw native text (including injection lines). Stripping SYSTEM overlays
    here over-counts ordinary image form pages as mystery and nets large
    true-approval damage; fee/bio candidate gating handles injection separately.
    """
    try:
        import fitz
    except Exception:
        return 0
    try:
        doc = fitz.open(pdf)
    except Exception:
        return 0
    n = 0
    try:
        for page in doc:
            native = page.get_text() or ""
            if not page.get_images() or len(native) >= 120:
                continue
            upper = native.upper()
            if any(
                k in upper
                for k in (
                    "FEE",
                    "RECEIPT",
                    "BIOMETRIC",
                    "B-13",
                    "OBSERVED",
                    "REGISTRY",
                    "INTAKE",
                    "SPONSOR",
                    "PASSPORT",
                    "FORM I",
                    "ADJUDICATOR",
                )
            ):
                continue
            n += 1
    finally:
        doc.close()
    return n


def _core_weak(record: dict[str, str]) -> bool:
    misses = 0
    if record.get("applicant_name") in {"unknown", ""}:
        misses += 1
    if record.get("species_code") == "unknown":
        misses += 1
    if record.get("home_world") == "unknown":
        misses += 1
    if record.get("sponsor_id") in {"SPN-0000", "unknown", ""}:
        misses += 1
    if record.get("arrival_date") in {"1900-01-01", "UNREADABLE", ""}:
        misses += 1
    if record.get("declared_purpose") == "unknown":
        misses += 1
    return misses >= 2


def _isotonic_confidence(raw: float) -> float:
    """Monotone identity-free recalibration of stratum confidence masses."""
    if raw in _ISO_MAP:
        return _ISO_MAP[raw]
    # Piecewise-constant nearest knot (strata masses are discrete).
    best = min(_ISO_MAP, key=lambda k: abs(k - raw))
    return _ISO_MAP[best]


def confidence_for(record: dict[str, str], *, explicit_quality: int = 0) -> float:
    if explicit_quality >= 100:
        return 0.995
    adj = record["adjudication"]
    letter = {"APPROVED": "A", "DENIED": "D", "NEEDS_REVIEW": "N"}[adj]
    fee = record["fee_status"] if record["fee_status"] in FEE_STATUSES else "unknown"
    visa = (
        record["visa_class"]
        if record["visa_class"] in {"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"}
        else "other"
    )
    rf = "flags" if record["risk_flags"] not in {"none", ""} else "none"
    weak = "weak" if _core_weak(record) else "ok"
    key = f"{letter}|{fee}|{visa}|{rf}|{weak}"
    raw = _CONF_FALLBACK[letter]
    if key in _CONF_STRATA:
        raw = _CONF_STRATA[key]
    else:
        # Soft fallbacks: try without weak, then without visa, then decision prior.
        for alt in (
            f"{letter}|{fee}|{visa}|{rf}|ok",
            f"{letter}|{fee}|other|{rf}|ok",
            f"{letter}|{fee}|other|none|ok",
            f"{letter}|unknown|other|{rf}|ok",
        ):
            if alt in _CONF_STRATA:
                raw = _CONF_STRATA[alt]
                break
    return _isotonic_confidence(raw)


def adjudicate_record(
    record: dict[str, str],
    explicit: str = "",
    unreadable: bool = False,
    *,
    flags_observed: bool = False,
) -> str:
    if explicit in ADJUDICATIONS:
        return explicit
    flags = set(record["risk_flags"].split("|")) - {"none", ""}
    visa = record["visa_class"]
    if flags & DISQUALIFYING_FLAGS:
        return "DENIED"
    if visa == "TRANSIT-7":
        return "DENIED"
    if visa != "DIP-1" and record["sponsor_id"] in REVOKED_SPONSORS:
        return "DENIED"
    if visa != "DIP-1" and record["home_world"] in NONDIP_EMBARGO:
        return "DENIED"
    if record["home_world"] in ALWAYS_EMBARGO_WORLDS:
        return "DENIED"
    if record["fee_status"] == "unpaid":
        return "DENIED"
    try:
        arrival = dt.date.fromisoformat(record["arrival_date"])
        if visa != "DIP-1" and (DEFAULT_RECEIPT_DATE - arrival).days > 180:
            return "DENIED"
    except ValueError:
        unreadable = True
    if unreadable or flags & REVIEW_FLAGS or record["fee_status"] == "unknown":
        return "NEEDS_REVIEW"
    # Optional strict visible-risk bar (strobl-style). Default off: full demotion
    # of unobserved-panel approvals nets ~-4 pts on public train (hurts true AP
    # more than it saves CFA). Enable with MIB_STRICT_FLAGS=1 for CFA-first runs.
    strict = os.environ.get("MIB_STRICT_FLAGS", "").strip() in {"1", "true", "yes"}
    if (
        strict
        and not flags_observed
        and record["risk_flags"] in {"none", ""}
    ):
        return "NEEDS_REVIEW"
    return "APPROVED"


def _try_review_recovery(
    record: dict[str, str],
    *,
    flags_observed: bool,
    damage: bool,
    native_len: int,
) -> bool:
    """Re-promote clean NEEDS_REVIEW → APPROVED when flags panel was visible."""
    if record["adjudication"] != "NEEDS_REVIEW":
        return False
    if record["fee_status"] not in {"paid", "waived"}:
        return False
    if record["fee_status"] == "waived" and record["visa_class"] != "DIP-1":
        return False
    if record["visa_class"] not in {"XW-1", "XW-2", "MED-3", "DIP-1"}:
        return False
    if record["risk_flags"] not in {"none", ""}:
        return False
    if not flags_observed:
        return False
    if _core_weak(record):
        return False
    if damage and native_len < 250:
        return False
    return True


def _fee_crop_fill(pdf: Path) -> str:
    """Hi-res Tess fee-band ensemble for UNKNOWN fees (no Rapid required)."""
    if os.environ.get("MIB_NO_FEE_CROP", "").strip() in {"1", "true", "yes"}:
        return ""
    try:
        from .rapid_ocr import page_looks_fee_candidate, tess_fee_crop_text
    except Exception:
        return ""
    try:
        import fitz
    except Exception:
        return ""
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ""
    chunks: list[str] = []
    budget = 0
    try:
        for page in doc:
            if budget >= 3:
                break
            if not page_looks_fee_candidate(page):
                continue
            text = tess_fee_crop_text(page, dpi=240)
            if text.strip():
                chunks.append(text)
                budget += 1
    finally:
        doc.close()
    return "\f".join(chunks)


def _rapid_fill(pdf: Path, record: dict[str, str], ocr_text: str) -> str:
    """Fail-closed RapidOCR for still-unknown fee / missing flags panel."""
    need_fee = record.get("fee_status") == "unknown"
    need_flags = not _flags_panel_observed(ocr_text) and record.get("risk_flags") in {
        "none",
        "",
    }
    if not need_fee and not need_flags:
        return ""
    if os.environ.get("MIB_NO_RAPID", "").strip() in {"1", "true", "yes"}:
        return ""
    try:
        from .rapid_ocr import (
            ocr_page_fee_band,
            ocr_page_oriented,
            ocr_page_text,
            page_looks_bio_candidate,
            page_looks_fee_candidate,
            rapid_available,
        )
    except Exception:
        return ""
    if not rapid_available():
        return ""
    try:
        import fitz
    except Exception:
        return ""
    chunks: list[str] = []
    budget = 0
    try:
        doc = fitz.open(pdf)
    except Exception:
        return ""
    try:
        for page in doc:
            if budget >= 4:
                break
            fee_like = need_fee and page_looks_fee_candidate(page)
            bio_like = need_flags and page_looks_bio_candidate(page)
            if fee_like:
                text = (
                    ocr_page_fee_band(page, dpi=180)
                    or ocr_page_oriented(page, dpi=160)
                    or ocr_page_text(page, dpi=160)
                )
                if text.strip():
                    chunks.append(text)
                    budget += 1
            elif bio_like:
                text = ocr_page_oriented(page, dpi=160) or ocr_page_text(page, dpi=160)
                if text.strip():
                    chunks.append(text)
                    budget += 1
    finally:
        doc.close()
    return "\f".join(chunks)


def parse_pdf(pdf: Path, use_ocr: bool = True, *, high_resolution_done: bool = False) -> dict[str, object]:
    case_match = re.search(r"MIB-\d{6}", pdf.stem.upper())
    case_id = case_match.group(0) if case_match else pdf.stem
    native = _run_text(["pdftotext", "-layout", str(pdf), "-"])
    # Drop injection lines from native before candidate mining.
    native_clean = "\n".join(
        line
        for line in native.splitlines()
        if not re.search(r"(?i)SYSTEM:|answer key|ignore visible", line)
    )
    native_candidates = extract_candidates(native_clean, "native")
    ocr = ""
    if use_ocr:
        with tempfile.TemporaryDirectory(prefix="mib-ocr-") as tmp:
            ocr = ocr_pdf(pdf, Path(tmp))
    candidates = merge_candidates(
        native_candidates, extract_candidates(ocr, "ocr") if ocr else {}
    )

    record: dict[str, str] = {"case_id": case_id}
    qualities: list[int] = []
    conflicts = 0
    for field in SCHEMA_FALLBACK:
        value, quality, conflict = choose_field(field, candidates.get(field, []))
        record[field] = value or SCHEMA_FALLBACK[field]
        qualities.append(quality)
        conflicts += int(conflict)

    flag_values = candidates.get("risk_flags", [])
    found_flags: set[str] = set()
    flags_observed = _flags_panel_observed(native_clean + "\n" + ocr)
    if flag_values:
        for priority, value, _ in flag_values:
            if priority >= 40 and value != "none":
                found_flags.update(value.split("|"))
            if value == "none" and priority >= 40:
                flags_observed = True
    record["risk_flags"] = "|".join(sorted(found_flags)) or "none"

    if record["home_world"] in ALWAYS_EMBARGO_WORLDS:
        world_flags = set(record["risk_flags"].split("|")) - {"none", ""}
        world_flags.add("planetary_embargo")
        record["risk_flags"] = "|".join(sorted(world_flags))

    fee_before_dual = record["fee_status"]
    # Dual-OCR recovery: Tess fee-crop ensemble then RapidOCR (strobl-style).
    if use_ocr and (
        record["fee_status"] == "unknown"
        or (not flags_observed and record["risk_flags"] in {"none", ""})
    ):
        enrich_chunks: list[str] = []
        if record["fee_status"] == "unknown":
            crop_text = _fee_crop_fill(pdf)
            if compact(crop_text):
                enrich_chunks.append(crop_text)
        rapid_text = _rapid_fill(pdf, record, native_clean + "\n" + ocr)
        if compact(rapid_text):
            enrich_chunks.append(rapid_text)
        if enrich_chunks:
            enrich = "\f".join(enrich_chunks)
            ocr = ocr + "\f" + enrich
            candidates = merge_candidates(
                candidates, extract_candidates(enrich, "ocr")
            )
            for field in SCHEMA_FALLBACK:
                value, quality, conflict = choose_field(
                    field, candidates.get(field, [])
                )
                record[field] = value or SCHEMA_FALLBACK[field]
                qualities.append(quality)
                conflicts += int(conflict)
            flag_values = candidates.get("risk_flags", [])
            found_flags = set()
            if flag_values:
                for priority, value, _ in flag_values:
                    if priority >= 40 and value != "none":
                        found_flags.update(value.split("|"))
                    if value == "none" and priority >= 40:
                        flags_observed = True
            if found_flags:
                record["risk_flags"] = "|".join(sorted(found_flags))
            flags_observed = flags_observed or _flags_panel_observed(enrich)

    fee_recovered = fee_before_dual == "unknown" and record["fee_status"] != "unknown"
    joined_text = native_clean + "\n" + ocr
    explicit, explicit_quality, explicit_conflict = choose(
        candidates.get("adjudication", [])
    )
    unreadable = bool(
        re.search(r"Arrival\s+Date\s*[: ]\s*UNREADABLE", native + "\n" + ocr, re.I)
    )
    record["adjudication"] = adjudicate_record(
        record, explicit, unreadable, flags_observed=flags_observed
    )

    # Gate: fee recovery alone must not create silent-stamp CFAs.
    if (
        fee_recovered
        and record["adjudication"] == "APPROVED"
        and not flags_observed
        and not _explicit_approved_signal(joined_text)
        and not explicit
    ):
        record["adjudication"] = "NEEDS_REVIEW"

    # Safe silent-stamp heuristic: mystery image page + no flags panel + no
    # explicit approval ⇒ demote APPROVED→REVIEW (catches hidden biohazard stamps
    # without blanket demotion of all unobserved-panel approvals).
    if (
        record["adjudication"] == "APPROVED"
        and record["risk_flags"] in {"none", ""}
        and not flags_observed
        and not explicit
        and not _explicit_approved_signal(joined_text)
        and _mystery_sparse_count(pdf) >= 1
    ):
        record["adjudication"] = "NEEDS_REVIEW"

    damage = bool(
        re.search(
            r"CUT OUT|WHITEOUT|WASHED OUT|UNREADABLE|REGISTRY LOST|PANEL MISSING|REDACTED",
            native + "\n" + ocr,
            re.I,
        )
    )
    damaged_approval = (
        not explicit
        and record["adjudication"] == "APPROVED"
        and (len(native) < 750 or damage)
    )
    severe_damage = bool(
        re.search(r"WHITEOUT|WASHED OUT|PANEL MISSING", native + "\n" + ocr, re.I)
    )
    weak_review = damaged_approval and (len(native) < 250 or severe_damage)
    if weak_review:
        record["adjudication"] = "NEEDS_REVIEW"

    if _try_review_recovery(
        record,
        flags_observed=flags_observed,
        damage=damage,
        native_len=len(native),
    ):
        record["adjudication"] = "APPROVED"

    confidence = confidence_for(record, explicit_quality=explicit_quality)
    if explicit_conflict or conflicts:
        confidence = max(0.35, confidence - 0.06)
    if record["adjudication"] == "APPROVED" and not flags_observed:
        # Should be rare after the bar; keep confidence humble if it happens.
        confidence = min(confidence, 0.55)
    record["confidence"] = round(float(confidence), 3)

    if (
        use_ocr
        and not high_resolution_done
        and "REDACTED" in ocr.upper()
        and record["risk_flags"] == "none"
        and record["confidence"] < 0.90
    ):
        with tempfile.TemporaryDirectory(prefix="mib-hires-") as tmp:
            high_resolution_ocr = ocr_pdf_high_resolution(pdf, Path(tmp))
        if compact(high_resolution_ocr):
            combined = high_resolution_ocr + "\f" + ocr
            native_candidates = extract_candidates(native_clean, "native")
            candidates = merge_candidates(
                native_candidates, extract_candidates(combined, "ocr")
            )
            for field in SCHEMA_FALLBACK:
                value, quality, conflict = choose_field(
                    field, candidates.get(field, [])
                )
                record[field] = value or SCHEMA_FALLBACK[field]
            flag_values = candidates.get("risk_flags", [])
            found_flags = set()
            flags_observed = _flags_panel_observed(combined)
            if flag_values:
                for priority, value, _ in flag_values:
                    if priority >= 40 and value != "none":
                        found_flags.update(value.split("|"))
                    if value == "none" and priority >= 40:
                        flags_observed = True
            record["risk_flags"] = "|".join(sorted(found_flags)) or "none"
            if record["home_world"] in ALWAYS_EMBARGO_WORLDS:
                world_flags = set(record["risk_flags"].split("|")) - {"none", ""}
                world_flags.add("planetary_embargo")
                record["risk_flags"] = "|".join(sorted(world_flags))
            explicit, explicit_quality, explicit_conflict = choose(
                candidates.get("adjudication", [])
            )
            record["adjudication"] = adjudicate_record(
                record, explicit, unreadable, flags_observed=flags_observed
            )
            if _try_review_recovery(
                record,
                flags_observed=flags_observed,
                damage=True,
                native_len=len(native),
            ):
                record["adjudication"] = "APPROVED"
            record["confidence"] = round(
                confidence_for(record, explicit_quality=explicit_quality), 3
            )

    return record


def process_one(pdf_path: str) -> dict:
    return parse_pdf(Path(pdf_path), use_ocr=True)
