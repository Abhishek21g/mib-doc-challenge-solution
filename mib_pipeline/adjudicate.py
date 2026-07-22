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

# Empirical P(correct | decision bucket) from public train — identity-free features only.
# Clipped away from {0,1} for stability off the training distribution.
_CONF = {
    "D|disq": 0.97,
    "D|transit": 0.96,
    "D|unpaid": 0.93,
    "D|revoked": 0.97,
    "D|other": 0.95,
    "R|fee": 0.57,
    "R|flag": 0.95,
    "R|core": 0.41,
    "R|other": 0.58,
    "A|feeunk": 0.97,
    "A|flags": 0.90,
    "A|clean": 0.71,
    "A|ocr": 0.65,
}


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


def _bucket(decision: str, reason: str, packet: PacketExtract) -> str:
    fee = field_value(packet, "fee_status", "unknown")
    visa = field_value(packet, "visa_class", "unknown")
    sponsor = field_value(packet, "sponsor_id", "SPN-0000")
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
        return "D|other"
    if decision == "NEEDS_REVIEW":
        if reason == "fee_unknown" or fee == "unknown":
            return "R|fee"
        if reason.startswith("review_flag") or (flags & REVIEW_FLAGS):
            return "R|flag"
        if reason in {"arrival_bad", "visa_missing", "visa_unknown", "fee_missing", "identity_gap"}:
            return "R|core"
        return "R|other"
    # APPROVED
    if reason == "manual_finding" and fee == "unknown":
        return "A|feeunk"
    if flags:
        return "A|flags"
    if packet.used_ocr and (
        packet.trusted_span_count < 20 or not packet.biometric_flags_observed
    ):
        return "A|ocr"
    return "A|clean"


def _confidence_for(decision: str, reason: str, packet: PacketExtract) -> float:
    return _CONF[_bucket(decision, reason, packet)]


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


def adjudicate(packet: PacketExtract) -> tuple[str, float, str]:
    """Return (adjudication, confidence, reason)."""
    visa = field_value(packet, "visa_class", "unknown")
    fee = field_value(packet, "fee_status", "unknown")
    sponsor = field_value(packet, "sponsor_id", "SPN-0000")
    home = field_value(packet, "home_world", "unknown")
    arrival_raw = field_value(packet, "arrival_date", "")
    arrival = _parse_date(arrival_raw if arrival_raw != "unknown" else None)
    flags = set(packet.risk_flags)
    issues = set(packet.evidence_issues)

    def finish(decision: str, reason: str) -> tuple[str, float, str]:
        return decision, _confidence_for(decision, reason, packet), reason

    if packet.manual_finding:
        return finish(packet.manual_finding, "manual_finding")

    hit = flags & DISQUALIFYING_FLAGS
    if hit:
        return finish("DENIED", f"disq_flag:{sorted(hit)[0]}")

    if home in ALWAYS_EMBARGO_WORLDS:
        return finish("DENIED", f"embargo_world:{home}")

    if home in NONDIP_EMBARGO_WORLDS and visa != "DIP-1":
        return finish("DENIED", "wolf_nondip")

    if visa == "TRANSIT-7":
        return finish("DENIED", "transit7")

    if sponsor in REVOKED_SPONSORS and visa != "DIP-1":
        return finish("DENIED", f"revoked:{sponsor}")

    if fee == "unpaid":
        return finish("DENIED", "unpaid")

    if visa != "DIP-1" and arrival is not None:
        if (DEFAULT_RECEIPT_DATE - arrival).days > 180:
            return finish("DENIED", "stale_arrival")

    if fee == "unknown":
        return finish("NEEDS_REVIEW", "fee_unknown")

    rev = flags & REVIEW_FLAGS
    if rev:
        return finish("NEEDS_REVIEW", f"review_flag:{sorted(rev)[0]}")

    if "arrival_unreadable" in issues or "arrival_missing" in issues:
        return finish("NEEDS_REVIEW", "arrival_bad")
    if "name_cut_out" in issues or "name_missing" in issues:
        return finish("NEEDS_REVIEW", "identity_gap")
    if "sponsor_conflict" in issues:
        return finish("NEEDS_REVIEW", "sponsor_conflict")
    if "visa_missing" in issues:
        return finish("NEEDS_REVIEW", "visa_missing")
    if "fee_missing" in issues:
        return finish("NEEDS_REVIEW", "fee_missing")
    if "image_heavy_uncertain" in issues and (
        "visa_missing" in issues or "fee_missing" in issues or "name_missing" in issues
    ):
        return finish("NEEDS_REVIEW", "uncertain_ocr")
    if visa == "unknown":
        return finish("NEEDS_REVIEW", "visa_unknown")
    if home == "unknown" and "name_missing" in issues:
        return finish("NEEDS_REVIEW", "core_fields_unknown")

    if visa != "DIP-1" and sponsor in {"SPN-0000", "unknown"} and "sponsor_conflict" not in issues:
        if "sponsor_id" not in {c.split(":")[0] for c in packet.conflicts}:
            if not packet.fields.get("sponsor_id"):
                return finish("NEEDS_REVIEW", "sponsor_missing")

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
    # Only demote when attestation is the sole policy anchor *and* biometric
    # risk panel was never observed (silent-stamp CFA pattern). Clean digital
    # packets without B-13 still approve when fee/visa are solid.
    if (
        attest_heavy
        and not intake_trusted
        and packet.used_ocr
        and not packet.biometric_flags_observed
        and "biometric" not in packet.docs_present
    ):
        return finish("NEEDS_REVIEW", "attest_without_intake")

    return finish("APPROVED", "clean")
