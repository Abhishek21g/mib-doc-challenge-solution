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

    # 1. Manual finding dominates when present
    if packet.manual_finding:
        conf = 0.93 if packet.fields.get("manual_finding", None) and packet.fields["manual_finding"].confidence >= 0.95 else 0.82
        return packet.manual_finding, conf, "manual_finding"

    # 2. Disqualifying flags
    hit = flags & DISQUALIFYING_FLAGS
    if hit:
        return "DENIED", 0.95, f"disq_flag:{sorted(hit)[0]}"

    # 3. Always-embargo worlds
    if home in ALWAYS_EMBARGO_WORLDS:
        return "DENIED", 0.94, f"embargo_world:{home}"

    # 4. Wolf-1061c non-DIP
    if home in NONDIP_EMBARGO_WORLDS and visa != "DIP-1":
        return "DENIED", 0.93, "wolf_nondip"

    # 5. TRANSIT-7
    if visa == "TRANSIT-7":
        return "DENIED", 0.96, "transit7"

    # 6. Revoked sponsor (DIP exempt)
    if sponsor in REVOKED_SPONSORS and visa != "DIP-1":
        return "DENIED", 0.94, f"revoked:{sponsor}"

    # 7. Unpaid fee
    if fee == "unpaid":
        return "DENIED", 0.95, "unpaid"

    # 8. Stale arrival (non-DIP)
    if visa != "DIP-1" and arrival is not None:
        if (DEFAULT_RECEIPT_DATE - arrival).days > 180:
            return "DENIED", 0.9, "stale_arrival"

    # 9. Unknown fee
    if fee == "unknown":
        return "NEEDS_REVIEW", 0.8, "fee_unknown"

    # 10. Review-only flags
    rev = flags & REVIEW_FLAGS
    if rev:
        return "NEEDS_REVIEW", 0.85, f"review_flag:{sorted(rev)[0]}"

    # 11. Evidence quality gates
    if "arrival_unreadable" in issues or "arrival_missing" in issues:
        return "NEEDS_REVIEW", 0.72, "arrival_bad"
    if "name_cut_out" in issues or "name_missing" in issues:
        return "NEEDS_REVIEW", 0.7, "identity_gap"
    if "sponsor_conflict" in issues:
        return "NEEDS_REVIEW", 0.75, "sponsor_conflict"
    if "visa_missing" in issues:
        return "NEEDS_REVIEW", 0.68, "visa_missing"
    if "fee_missing" in issues:
        return "NEEDS_REVIEW", 0.68, "fee_missing"
    if "image_heavy_uncertain" in issues and (
        "visa_missing" in issues or "fee_missing" in issues or "name_missing" in issues
    ):
        return "NEEDS_REVIEW", 0.55, "uncertain_ocr"
    if visa == "unknown":
        return "NEEDS_REVIEW", 0.6, "visa_unknown"
    if home == "unknown" and "name_missing" in issues:
        return "NEEDS_REVIEW", 0.6, "core_fields_unknown"

    # Missing sponsor id on non-DIP with otherwise complete packet → review
    if visa != "DIP-1" and sponsor in {"SPN-0000", "unknown"} and "sponsor_conflict" not in issues:
        # Only force review when sponsor truly absent from form evidence
        if "sponsor_id" not in {c.split(":")[0] for c in packet.conflicts}:
            if not packet.fields.get("sponsor_id"):
                return "NEEDS_REVIEW", 0.65, "sponsor_missing"

    return "APPROVED", 0.88, "clean"
