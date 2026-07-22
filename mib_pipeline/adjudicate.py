from __future__ import annotations

from datetime import date, datetime

from .constants import (
    ALWAYS_EMBARGO_WORLDS,
    DEFAULT_RECEIPT_DATE,
    DISQUALIFYING_FLAGS,
    NONDIP_EMBARGO_WORLDS,
    REVIEW_FLAGS,
    REVOKED_SPONSORS,
)
from .extract import PacketExtract

# Empirical P(correct | identity-free reason/feature bucket) from public train.
# Tuned toward strobl/goleffect-style high deny/review reliability; clipped.
_CONF = {
    "D|disq": 0.99,
    "D|transit": 0.98,
    "D|unpaid": 0.97,
    "D|revoked": 0.98,
    "D|stale": 0.96,
    "D|other": 0.97,
    "R|fee|DIP-1": 0.68,
    "R|fee|XW-1": 0.70,
    "R|fee|XW-2": 0.66,
    "R|fee|MED-3": 0.60,
    "R|fee|unknown": 0.58,
    "R|fee|other": 0.62,
    "R|flag": 0.92,
    "R|arr": 0.68,
    "R|visa": 0.55,
    "R|name": 0.60,
    "R|spn": 0.55,
    "R|silent": 0.70,
    "R|damage": 0.72,
    "R|soft|DIP-1": 0.70,
    "R|soft|XW-1": 0.64,
    "R|soft|XW-2": 0.60,
    "R|soft|MED-3": 0.62,
    "R|soft|other": 0.64,
    "R|other": 0.66,
    "A|manual": 0.995,
    "A|feeunk": 0.88,
    "A|flags": 0.84,
    "A|paid|DIP-1": 0.80,
    "A|paid|XW-1": 0.76,
    "A|paid|XW-2": 0.72,
    "A|paid|MED-3": 0.74,
    "A|paid|other": 0.74,
    "A|waived|DIP-1": 0.86,
    "A|waived|XW-1": 0.50,
    "A|waived|XW-2": 0.32,
    "A|waived|MED-3": 0.50,
    "A|waived|other": 0.48,
    "A|ocr": 0.70,
    "A|clean": 0.74,
    "A|recover": 0.78,
}

_WEAK_REVIEW_REASONS = frozenset(
    {
        "attest_without_intake",
        "uncertain_ocr",
        "silent_risk_gap",
        "damaged_packet",
    }
)
_RECOVERABLE_VISAS = frozenset({"XW-1", "XW-2", "MED-3", "DIP-1"})


def _parse_date(value: str | None) -> date | None:
    if not value or value == "UNREADABLE":
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def field_value(packet: PacketExtract, key: str, default: str = "unknown") -> str:
    hit = packet.fields.get(key)
    if not hit:
        return default
    return hit.value


def _core_complete(packet: PacketExtract) -> bool:
    name = field_value(packet, "applicant_name", "unknown")
    if name in {"unknown", "[NAME CUT OUT]"}:
        return False
    if field_value(packet, "arrival_date", "1900-01-01") in {
        "",
        "unknown",
        "1900-01-01",
        "UNREADABLE",
    }:
        return False
    if field_value(packet, "species_code", "unknown") == "unknown":
        return False
    if field_value(packet, "home_world", "unknown") == "unknown":
        return False
    sponsor = field_value(packet, "sponsor_id", "SPN-0000")
    if sponsor in {"SPN-0000", "unknown"} or not sponsor.startswith("SPN-"):
        return False
    return True


def _packet_damage_score(packet: PacketExtract) -> int:
    """Cheap damage heuristic (goleffect-style) — identity-free."""
    score = 0
    if packet.trusted_span_count < 250:
        score += 1
    if packet.trusted_span_count < 80:
        score += 2
    if packet.used_ocr:
        score += 1
    if "image_heavy_uncertain" in packet.evidence_issues:
        score += 2
    joined = " ".join(p.trusted_text for p in packet.pages).upper()
    for needle in (
        "CUT OUT",
        "WHITEOUT",
        "WASHED OUT",
        "UNREADABLE",
        "PANEL MISSING",
        "REDACTED",
        "REGISTRY LOST",
    ):
        if needle in joined:
            score += 2
    return score


def _try_review_approve_recovery(
    packet: PacketExtract,
    reason: str,
) -> tuple[str, str] | None:
    """Recover NEEDS_REVIEW → APPROVED for complete clean packets.

    Never recovers fee-unknown, review flags, silent_risk_gap, or incomplete cores.
    CFA guard requires biometric panel observed OR digital-clean trusted intake.
    """
    if reason not in {"attest_without_intake", "uncertain_ocr"}:
        return None
    visa = field_value(packet, "visa_class", "unknown")
    fee = field_value(packet, "fee_status", "unknown")
    if visa not in _RECOVERABLE_VISAS:
        return None
    if fee not in {"paid", "waived"}:
        return None
    if fee == "waived" and visa != "DIP-1":
        return None
    if packet.risk_flags & (DISQUALIFYING_FLAGS | REVIEW_FLAGS):
        return None
    if packet.risk_flags:
        return None
    if not _core_complete(packet):
        return None
    if _packet_damage_score(packet) >= 4:
        return None

    bio_ok = packet.biometric_flags_observed or "biometric" in packet.docs_present
    digital_clean = (
        fee == "paid"
        and visa in {"XW-1", "DIP-1"}
        and "intake" in packet.docs_present
        and packet.trusted_span_count >= 20
    )
    dip_waived = (
        fee == "waived"
        and visa == "DIP-1"
        and "fee_receipt" in packet.docs_present
        and packet.trusted_span_count >= 15
    )
    # XW-1 complete multisource packet (strobl-inspired, our features only).
    xw1_complete = (
        visa == "XW-1"
        and fee == "paid"
        and "intake" in packet.docs_present
        and "sponsor_letter" in packet.docs_present
        and packet.trusted_span_count >= 25
        and (bio_ok or packet.biometric_flags_observed)
    )
    if not (bio_ok or digital_clean or dip_waived or xw1_complete):
        return None
    return "APPROVED", f"recover:{reason}"


def _try_silent_stamp_demotion(packet: PacketExtract, reason: str) -> tuple[str, str] | None:
    """APPROVED → REVIEW only for severe visible damage with no biometric panel.

    Broad demotion mass-converts true approvals to review and nets negative on
    public train. Silent stamps remain an unsolved CFA class without CV.
    """
    if reason == "manual_finding" or reason.startswith("recover"):
        return None
    if packet.biometric_flags_observed or "biometric" in packet.docs_present:
        return None
    if packet.risk_flags & DISQUALIFYING_FLAGS:
        return None
    joined = " ".join(p.trusted_text for p in packet.pages).upper()
    severe = any(
        n in joined
        for n in ("WHITEOUT", "WASHED OUT", "PANEL MISSING", "RISK PANEL MISSING")
    )
    if severe and packet.used_ocr and packet.trusted_span_count < 25:
        return "NEEDS_REVIEW", "silent_risk_gap"
    return None


def _bucket(decision: str, reason: str, packet: PacketExtract) -> str:
    fee = field_value(packet, "fee_status", "unknown")
    visa = field_value(packet, "visa_class", "unknown")
    flags = set(packet.risk_flags)
    if decision == "DENIED":
        if reason.startswith("disq_flag") or reason.startswith("embargo") or reason == "wolf_nondip":
            return "D|disq"
        if reason == "transit7":
            return "D|transit"
        if reason == "unpaid":
            return "D|unpaid"
        if reason.startswith("revoked"):
            return "D|revoked"
        if reason == "stale_arrival":
            return "D|stale"
        return "D|other"
    if decision == "NEEDS_REVIEW":
        if reason == "silent_risk_gap":
            return "R|silent"
        if reason == "damaged_packet":
            return "R|damage"
        if reason == "fee_unknown" or fee == "unknown":
            if visa in {"XW-1", "XW-2", "DIP-1", "MED-3", "unknown"}:
                return f"R|fee|{visa}"
            return "R|fee|other"
        if reason.startswith("review_flag") or (flags & REVIEW_FLAGS):
            return "R|flag"
        if reason == "arrival_bad":
            return "R|arr"
        if reason in {"visa_missing", "visa_unknown"}:
            return "R|visa"
        if reason == "identity_gap":
            return "R|name"
        if reason == "sponsor_missing":
            return "R|spn"
        if reason in _WEAK_REVIEW_REASONS or reason.startswith("recover"):
            if visa in {"XW-1", "XW-2", "DIP-1", "MED-3"}:
                return f"R|soft|{visa}"
            return "R|soft|other"
        return "R|other"
    if reason == "manual_finding":
        return "A|manual"
    if reason.startswith("recover"):
        return "A|recover"
    if fee == "unknown":
        return "A|feeunk"
    if flags:
        return "A|flags"
    if fee in {"paid", "waived"} and visa in {"XW-1", "XW-2", "DIP-1", "MED-3"}:
        return f"A|{fee}|{visa}"
    if fee == "paid":
        return "A|paid|other"
    if fee == "waived":
        return "A|waived|other"
    if packet.used_ocr and (
        packet.trusted_span_count < 20 or not packet.biometric_flags_observed
    ):
        return "A|ocr"
    return "A|clean"


def _confidence_for(decision: str, reason: str, packet: PacketExtract) -> float:
    key = _bucket(decision, reason, packet)
    if key in _CONF:
        return _CONF[key]
    if decision == "DENIED":
        return 0.97
    if decision == "NEEDS_REVIEW":
        return 0.72
    return 0.74


def build_record(packet: PacketExtract) -> dict[str, str | float]:
    adjudication, confidence, reason = adjudicate(packet)
    flags = sorted(packet.risk_flags)
    risk_flags = "|".join(flags) if flags else "none"

    sponsor = field_value(packet, "sponsor_id", "SPN-0000")
    if not sponsor.startswith("SPN-"):
        sponsor = "SPN-0000"

    arrival = field_value(packet, "arrival_date", "1900-01-01")
    if arrival == "UNREADABLE":
        arrival = "1900-01-01"

    fee = field_value(packet, "fee_status", "unknown")
    if fee not in {"paid", "waived", "unpaid", "unknown"}:
        fee = "unknown"

    return {
        "case_id": packet.case_id,
        "applicant_name": field_value(packet, "applicant_name", "unknown"),
        "species_code": field_value(packet, "species_code", "unknown"),
        "home_world": field_value(packet, "home_world", "unknown"),
        "visa_class": field_value(packet, "visa_class", "unknown"),
        "sponsor_id": sponsor,
        "arrival_date": arrival,
        "declared_purpose": field_value(packet, "declared_purpose", "unknown"),
        "risk_flags": risk_flags,
        "fee_status": fee,
        "adjudication": adjudication,
        "confidence": confidence,
        "_reason": reason,
    }


def _baseline_decision(packet: PacketExtract) -> tuple[str, str]:
    """Policy tree without confidence or recovery overrides."""
    visa = field_value(packet, "visa_class", "unknown")
    fee = field_value(packet, "fee_status", "unknown")
    sponsor = field_value(packet, "sponsor_id", "SPN-0000")
    home = field_value(packet, "home_world", "unknown")
    arrival_raw = field_value(packet, "arrival_date", "")
    arrival = _parse_date(arrival_raw if arrival_raw != "unknown" else None)
    flags = set(packet.risk_flags)
    issues = set(packet.evidence_issues)

    if packet.manual_finding:
        return packet.manual_finding, "manual_finding"

    hit = flags & DISQUALIFYING_FLAGS
    if hit:
        return "DENIED", f"disq_flag:{sorted(hit)[0]}"

    if home in ALWAYS_EMBARGO_WORLDS:
        return "DENIED", f"embargo_world:{home}"

    if home in NONDIP_EMBARGO_WORLDS and visa != "DIP-1":
        return "DENIED", "wolf_nondip"

    if visa == "TRANSIT-7":
        return "DENIED", "transit7"

    if sponsor in REVOKED_SPONSORS and visa != "DIP-1":
        return "DENIED", f"revoked:{sponsor}"

    if fee == "unpaid":
        return "DENIED", "unpaid"

    if visa != "DIP-1" and arrival is not None:
        if (DEFAULT_RECEIPT_DATE - arrival).days > 180:
            return "DENIED", "stale_arrival"

    if fee == "unknown":
        return "NEEDS_REVIEW", "fee_unknown"

    rev = flags & REVIEW_FLAGS
    if rev:
        return "NEEDS_REVIEW", f"review_flag:{sorted(rev)[0]}"

    if "arrival_unreadable" in issues or "arrival_missing" in issues:
        return "NEEDS_REVIEW", "arrival_bad"
    if "name_cut_out" in issues or "name_missing" in issues:
        return "NEEDS_REVIEW", "identity_gap"
    if "sponsor_conflict" in issues:
        return "NEEDS_REVIEW", "sponsor_conflict"
    if "visa_missing" in issues:
        return "NEEDS_REVIEW", "visa_missing"
    if "fee_missing" in issues:
        return "NEEDS_REVIEW", "fee_missing"
    if "image_heavy_uncertain" in issues and (
        "visa_missing" in issues or "fee_missing" in issues or "name_missing" in issues
    ):
        return "NEEDS_REVIEW", "uncertain_ocr"
    if visa == "unknown":
        return "NEEDS_REVIEW", "visa_unknown"
    if home == "unknown" and "name_missing" in issues:
        return "NEEDS_REVIEW", "core_fields_unknown"

    if visa != "DIP-1" and sponsor in {"SPN-0000", "unknown"} and "sponsor_conflict" not in issues:
        if "sponsor_id" not in {c.split(":")[0] for c in packet.conflicts}:
            if not packet.fields.get("sponsor_id"):
                return "NEEDS_REVIEW", "sponsor_missing"

    intake_keys = ("species_code", "home_world", "arrival_date")
    intake_trusted = all(
        (hit := packet.fields.get(k)) is not None
        and hit.source in {"intake", "adjudicator_note", "registry", "biometric"}
        for k in intake_keys
    )
    attest_heavy = any(
        (hit := packet.fields.get(k)) is not None and hit.source in {"sponsor_letter", "ocr"}
        for k in ("visa_class", "sponsor_id")
    )
    if (
        attest_heavy
        and not intake_trusted
        and packet.used_ocr
        and not packet.biometric_flags_observed
        and "biometric" not in packet.docs_present
    ):
        return "NEEDS_REVIEW", "attest_without_intake"

    if _packet_damage_score(packet) >= 5 and not packet.biometric_flags_observed:
        return "NEEDS_REVIEW", "damaged_packet"

    return "APPROVED", "clean"


def adjudicate(packet: PacketExtract) -> tuple[str, float, str]:
    """Return (adjudication, confidence, reason)."""
    decision, reason = _baseline_decision(packet)
    if decision == "NEEDS_REVIEW":
        recovered = _try_review_approve_recovery(packet, reason)
        if recovered is not None:
            decision, reason = recovered
    if decision == "APPROVED":
        demoted = _try_silent_stamp_demotion(packet, reason)
        if demoted is not None:
            decision, reason = demoted
    return decision, _confidence_for(decision, reason, packet), reason
