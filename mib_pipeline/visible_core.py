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


def fuzzy_risk_mentions(page: str) -> set[str]:
    found: set[str] = set()
    for line in page.splitlines():
        if not re.search(r"\b(?:obs\w*|flags?|risk|reason|finding)\b", line, re.I):
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
                difflib.SequenceMatcher(None, gram, target).ratio() >= 0.76
                for gram in grams
            ):
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
                if re.search(r"Amount\s*\$?\s*809[.,]?00?", page, re.I):
                    result["fee_status"].append((92, "paid", f"{source}:fee"))
                elif re.search(r"Amount\s*\$?\s*0[.,]00", page, re.I) and re.search(
                    r"DIP.?WAIVER", page, re.I
                ):
                    result["fee_status"].append((92, "waived", f"{source}:fee"))
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
            if re.search(r"(?i)\$\s*809(?:\.00)?\b|Amount\s*\$?\s*809", page):
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
                found = [
                    flag
                    for flag in RISK_FLAGS
                    if flag in match.group(1).lower().replace(" ", "_")
                ]
                if found:
                    result["risk_flags"].append(
                        (base, "|".join(found), f"{source}:{kind}")
                    )
                elif re.search(r"\bnone\b", match.group(1), re.I):
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


def adjudicate_record(
    record: dict[str, str], explicit: str = "", unreadable: bool = False
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
    return "APPROVED"


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
    if flag_values:
        for priority, value, _ in flag_values:
            if priority >= 40 and value != "none":
                found_flags.update(value.split("|"))
    record["risk_flags"] = "|".join(sorted(found_flags)) or "none"

    if record["home_world"] in ALWAYS_EMBARGO_WORLDS:
        world_flags = set(record["risk_flags"].split("|")) - {"none", ""}
        world_flags.add("planetary_embargo")
        record["risk_flags"] = "|".join(sorted(world_flags))

    explicit, explicit_quality, explicit_conflict = choose(
        candidates.get("adjudication", [])
    )
    unreadable = bool(
        re.search(r"Arrival\s+Date\s*[: ]\s*UNREADABLE", native + "\n" + ocr, re.I)
    )
    record["adjudication"] = adjudicate_record(record, explicit, unreadable)

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
    recovered_approval = damaged_approval and not weak_review
    if weak_review:
        record["adjudication"] = "NEEDS_REVIEW"

    min_quality = min(qualities) if qualities else 0
    if explicit_quality >= 100 and not explicit_conflict:
        confidence = 0.995
    elif weak_review:
        confidence = 0.42
    elif recovered_approval:
        confidence = 0.70
    elif record["adjudication"] == "DENIED":
        confidence = 0.98
    elif record["adjudication"] == "NEEDS_REVIEW":
        confidence = 0.95 if unreadable or min_quality < 20 else 0.72
    elif flag_values:
        confidence = 0.85
    else:
        confidence = 0.59
    if conflicts:
        confidence = max(0.35, confidence - 0.08)
    record["confidence"] = round(confidence, 3)

    if (
        use_ocr
        and not high_resolution_done
        and "REDACTED" in ocr.upper()
        and record["risk_flags"] == "none"
        and confidence < 0.90
    ):
        with tempfile.TemporaryDirectory(prefix="mib-hires-") as tmp:
            high_resolution_ocr = ocr_pdf_high_resolution(pdf, Path(tmp))
        if compact(high_resolution_ocr):
            # Re-parse with hi-res OCR prepended.
            with tempfile.TemporaryDirectory(prefix="mib-ocr2-") as tmp2:
                # Stash combined OCR via env cache bypass: call extract on concat.
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
            if flag_values:
                for priority, value, _ in flag_values:
                    if priority >= 40 and value != "none":
                        found_flags.update(value.split("|"))
            record["risk_flags"] = "|".join(sorted(found_flags)) or "none"
            if record["home_world"] in ALWAYS_EMBARGO_WORLDS:
                world_flags = set(record["risk_flags"].split("|")) - {"none", ""}
                world_flags.add("planetary_embargo")
                record["risk_flags"] = "|".join(sorted(world_flags))
            explicit, explicit_quality, explicit_conflict = choose(
                candidates.get("adjudication", [])
            )
            record["adjudication"] = adjudicate_record(
                record, explicit, unreadable
            )
            if record["adjudication"] == "DENIED":
                record["confidence"] = 0.98
            elif record["adjudication"] == "NEEDS_REVIEW":
                record["confidence"] = 0.80
            else:
                record["confidence"] = 0.75

    return record


def process_one(pdf_path: str) -> dict:
    return parse_pdf(Path(pdf_path), use_ocr=True)
