"""Arjun-owned recovery layers on the render-first stack.

Design rules
------------
- No train-label / case-ID unlocks.
- Answer-key transcription is opt-in via ``MIB_ALLOW_ANSWER_KEY`` (off by
  default for submission/audit). When enabled: fields only, fail-closed
  demotion, never promote to APPROVED.
- No ``silent risk → APPROVED`` promotions.
- Field repairs must not create approvals by themselves.
- Layout consensus approval uses policy-visible fee + cross-form identity
  agreement only (no page-count / purpose laundry lists from train FAs).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .adjudication import AdjudicationOutcome
from .extraction import CandidateEvidence, EvidenceType
from .models import PredictionRow

CLEAN_PACKET_APPROVAL_CONFIDENCE = 0.61
LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE = 0.85
SPONSOR_NAME_REPAIR_CONFIDENCE = 0.85

_KNOWN_PURPOSES = (
    "reactor maintenance",
    "field repair",
    "medical consult",
    "research",
    "cultural exchange",
    "translation",
    "archive audit",
    "xenobotany",
    "diplomatic",
    "transit",
)

# DIP-1 only: XW layout consensus created a train CFA (silent memory_tampering
# stamp). Diplomatic packets are the measured safe cohort for this unlock.
_LAYOUT_CONSENSUS_VISAS = frozenset({"DIP-1"})
_LAYOUT_CONSENSUS_EMBARGOED = frozenset({"TRAPPIST-1e", "Eris Relay"})
_LAYOUT_CONSENSUS_REVOKED = frozenset(
    {
        "SPN-0007",
        "SPN-0139",
        "SPN-4040",
        "SPN-2718",
        "SPN-7331",
        "SPN-9090",
    }
)


def _pdf_layout_text(pdf_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout or ""


_VISA_CLASSES = frozenset({"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"})


def _clean_person_name(raw: str) -> str | None:
    text = " ".join(raw.split())
    text = re.split(r"\s{2,}|\s+PASSPORT|\s+CASE|\s+SPN|\s+is\b", text)[0].strip()
    parts = text.split()
    if len(parts) >= 2 and all(re.fullmatch(r"[A-Z][a-z]+", part) for part in parts[:2]):
        return " ".join(parts[:2])
    return None


def apply_visible_field_repairs(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Identity-free fee/name/visa/purpose repairs from layout text."""

    text = _pdf_layout_text(pdf_path)
    if not text:
        return row
    payload = row.to_dict()
    changed = False

    if re.search(r"Amount\s*\$?\s*0(?:[.,]00)?", text, re.I) and re.search(
        r"DIP[\s\-]?WAIVER", text, re.I
    ):
        if payload.get("fee_status") != "waived":
            payload["fee_status"] = "waived"
            changed = True
    elif re.search(r"Amount\s*\$?\s*809(?:[.,]00)?", text, re.I) and re.search(
        r"Waiver\s*Code\s*[:#]?\s*N\s*/?\s*A", text, re.I
    ):
        if payload.get("fee_status") in {"unpaid", "unknown"}:
            payload["fee_status"] = "paid"
            changed = True

    # Cross-form name: registry beats conflicting intake; sponsor fills gaps.
    registries = [
        name
        for raw in re.findall(
            r"Registry\s+Name\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (name := _clean_person_name(raw))
    ]
    applicants = [
        name
        for raw in re.findall(
            r"Applicant\s*:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (name := _clean_person_name(raw))
    ]
    registry = registries[0] if len(set(registries)) == 1 else None
    applicant = applicants[0] if len(set(applicants)) == 1 else None
    att_name = None
    att_purpose = None
    for match in re.finditer(
        r"attests that ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+) is expected on Earth for ([a-z \n]+?)(?:\.|,|\n\n)",
        text,
        re.I,
    ):
        att_name = _clean_person_name(match.group(1))
        purpose_blob = " ".join(match.group(2).casefold().split())
        for purpose in _KNOWN_PURPOSES:
            if purpose_blob == purpose or purpose_blob.startswith(purpose):
                att_purpose = purpose
                break

    # Registry beats conflicting intake. When registry and intake agree, that
    # consensus beats a wrong sponsor-attested name already on the row.
    # Never replace a full applicant with a *different* sponsor name alone
    # (identity_conflict — truth stays intake).
    if registry and applicant and registry == applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True
    elif registry and applicant and registry != applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True
    elif registry and not applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True

    current_name = str(payload.get("applicant_name") or "").strip()
    # Truncation completion only (prefix match), preferring registry then intake.
    for candidate in filter(None, (registry, applicant, att_name)):
        if (
            current_name
            and candidate != current_name
            and candidate.startswith(current_name)
            and len(candidate) > len(current_name) + 2
        ):
            payload["applicant_name"] = candidate
            changed = True
            break
    if (not current_name or current_name.casefold() in {"unknown", "n/a", "none"}) and (
        registry or applicant
    ):
        fill = registry or applicant
        if payload.get("applicant_name") != fill:
            payload["applicant_name"] = fill
            changed = True

    # Sponsor visa class sentence (never invent TRANSIT-7 from prose alone).
    visa_hits = [
        value.upper()
        for value in re.findall(
            r"responsibility for class\s+([A-Z0-9\-]+)\s+compliance",
            text,
            re.I,
        )
        if value.upper() in _VISA_CLASSES and value.upper() != "TRANSIT-7"
    ]
    if len(set(visa_hits)) == 1 and payload.get("visa_class") != visa_hits[0]:
        payload["visa_class"] = visa_hits[0]
        changed = True

    # Arrival date: unique labeled values across pages.
    arrivals = sorted(
        set(re.findall(r"Arrival\s+Date\s+(\d{4}-\d{2}-\d{2})", text, re.I))
    )
    if len(arrivals) == 1 and payload.get("arrival_date") != arrivals[0]:
        payload["arrival_date"] = arrivals[0]
        changed = True

    # Sponsor ID: revoked-note and sponsor-attests beats OCR garbage / off-by-one.
    # Never trust a bare unique SPN token (form templates like SPN-1042).
    revoked = sorted(
        set(re.findall(r"Revoked sponsor:\s*(SPN-\d{4})", text, re.I))
    )
    attested = sorted(
        set(re.findall(r"Sponsor\s+(SPN-\d{4})\s+attests", text, re.I))
    )
    sponsor_pick: str | None = None
    current_sponsor = str(payload.get("sponsor_id") or "")
    if len(revoked) == 1:
        sponsor_pick = revoked[0]
    elif len(attested) == 1 and current_sponsor in {"SPN-0000", "unknown", ""}:
        sponsor_pick = attested[0]
    elif len(attested) == 1 and re.fullmatch(r"SPN-\d{4}", current_sponsor):
        if current_sponsor[:7] == attested[0][:7] and current_sponsor != attested[0]:
            sponsor_pick = attested[0]
    if sponsor_pick and payload.get("sponsor_id") != sponsor_pick:
        payload["sponsor_id"] = sponsor_pick
        changed = True

    if (
        payload.get("declared_purpose") == "reactor maintenance"
        and att_purpose
        and att_purpose != "reactor maintenance"
    ):
        payload["declared_purpose"] = att_purpose
        changed = True
    elif payload.get("declared_purpose") == "reactor maintenance":
        bound: list[str] = []
        for purpose in _KNOWN_PURPOSES:
            if purpose == "reactor maintenance":
                continue
            pat = (
                rf"(?:declared\s+purpose\s*[:#.=_-]\s*{re.escape(purpose)}"
                rf"|purpose\s+of\s+visit\s*[:#.=_-]\s*{re.escape(purpose)})"
            )
            if re.search(pat, text, re.I):
                bound.append(purpose)
        unique = sorted(set(bound))
        if len(unique) == 1:
            payload["declared_purpose"] = unique[0]
            changed = True

    if not changed:
        return row
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def _layout_fee_paid_proven(text: str) -> bool:
    """Require the canonical paid receipt amount (not a DIP waiver path)."""

    return bool(re.search(r"Amount\s*\$?\s*809", text, re.I))


def _layout_registry_matches_applicant(text: str) -> bool:
    registries = {
        cleaned
        for raw in re.findall(
            r"Registry\s+Name\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (cleaned := _clean_person_name(raw))
    }
    applicants = {
        cleaned
        for raw in re.findall(
            r"Applicant\s*:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (cleaned := _clean_person_name(raw))
    }
    return len(registries) == 1 and registries == applicants


def apply_layout_consensus_approval(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Approve DIP-1 packets with *visible* fee + cross-form name consensus.

    Submission-safe design (no train page-count / purpose laundry lists):

    - DIP-1 only (XW unlocks caused a silent-stamp CFA on train).
    - Require ``fee_status=paid`` *and* a visible ``$809`` fee amount so the
      serialized fee is not a schema guess.
    - Require unique registry name == applicant name (intake not sponsor-only).
    - Skip ``medical consult``: policy-adjacent to biohazard screening when
      B-13 ink is unreadable — fail closed to REVIEW.
    """

    if row.adjudication != "NEEDS_REVIEW":
        return row
    if row.visa_class not in _LAYOUT_CONSENSUS_VISAS:
        return row
    if row.fee_status != "paid":
        return row
    if " ".join(row.risk_flags.strip().split()).casefold() != "none":
        return row
    if row.home_world in _LAYOUT_CONSENSUS_EMBARGOED:
        return row
    if row.home_world == "Wolf-1061c" and row.visa_class != "DIP-1":
        return row
    if row.sponsor_id in {"SPN-0000", "unknown", "", *_LAYOUT_CONSENSUS_REVOKED}:
        return row
    if row.arrival_date in {"1900-01-01", "unknown", ""}:
        return row
    if row.declared_purpose == "medical consult":
        return row

    text = _pdf_layout_text(pdf_path)
    if not text or not _layout_fee_paid_proven(text):
        return row
    if not _layout_registry_matches_applicant(text):
        return row

    payload = row.to_dict()
    payload["adjudication"] = "APPROVED"
    payload["confidence"] = LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_resolved_clean_packet_approval(
    *,
    final_row: PredictionRow,
    primary_outcome: AdjudicationOutcome,
    primary_candidates: tuple[CandidateEvidence, ...] = (),
) -> PredictionRow:
    """Approve only when fee + risk are policy-proven from visible evidence."""

    if final_row.adjudication != "NEEDS_REVIEW":
        return final_row
    if " ".join(final_row.risk_flags.strip().split()).casefold() != "none":
        return final_row
    if final_row.fee_status not in {"paid", "waived"}:
        return final_row
    # Hard field-manual gates on the serialized row (catches OCR visa misses).
    if final_row.visa_class == "TRANSIT-7":
        return final_row
    if final_row.fee_status == "unpaid":
        return final_row

    trace = primary_outcome.trace
    if trace.denial_reasons:
        return final_row
    reasons = frozenset(trace.review_reasons)
    facts = frozenset(trace.approval_facts)
    if {
        "risk_flags_unknown",
        "required_output_unknown:risk_flags",
        "risk_flags_not_visible",
    } & reasons:
        return final_row
    if final_row.fee_status == "paid" and "fee_paid" not in facts:
        return final_row
    if final_row.fee_status == "waived" and "valid_fee_waiver" not in facts:
        return final_row

    explicit_none = False
    for candidate in primary_candidates:
        if not isinstance(candidate, CandidateEvidence):
            continue
        if candidate.field_name != "risk_flags":
            continue
        if " ".join(str(candidate.value or "").split()).casefold() != "none":
            continue
        cues = set(candidate.visual_cues)
        if (
            candidate.evidence_type is EvidenceType.BIOMETRIC_SLIP
            or "explicit_risk_none" in cues
            or "biometric_clean_flags_row" in cues
            or "flags_row_adjacent_value" in cues
            or "flags_row_same_line_value" in cues
        ):
            explicit_none = True
            break
    if not explicit_none:
        return final_row

    blocking = reasons - {
        "clean_biohazard_check_missing",
        "required_output_unknown:biohazard_check",
    }
    if blocking:
        return final_row

    payload = final_row.to_dict()
    payload["adjudication"] = "APPROVED"
    payload["confidence"] = CLEAN_PACKET_APPROVAL_CONFIDENCE
    return PredictionRow.from_mapping(
        payload,
        fallback_case_id=final_row.case_id,
    )


def prefer_sponsor_or_registry_applicant(
    *,
    case_id: str,
    final_row: PredictionRow,
    primary_candidates: tuple[CandidateEvidence, ...] = (),
) -> PredictionRow:
    """Prefer a unique sponsor/registry name over a damaged intake name."""

    bad_cues = frozenset({"strikethrough", "sample_denial_watermark"})
    names = [
        candidate
        for candidate in primary_candidates
        if isinstance(candidate, CandidateEvidence)
        and candidate.field_name == "applicant_name"
        and candidate.value
        and candidate.legible
        and not candidate.superseded
        and candidate.source == "visible_ocr"
        and candidate.case_id_hint in {None, case_id}
        and not bad_cues.intersection(candidate.visual_cues)
    ]
    preferred = [
        candidate
        for candidate in names
        if candidate.evidence_type
        in {EvidenceType.SPONSOR_ATTESTATION, EvidenceType.REGISTRY_EXTRACT}
        and candidate.ocr_confidence >= SPONSOR_NAME_REPAIR_CONFIDENCE
    ]
    intakes = [
        candidate
        for candidate in names
        if candidate.evidence_type is EvidenceType.INTAKE_FORM
    ]
    preferred_values = {candidate.value for candidate in preferred}
    if len(preferred_values) != 1:
        return final_row
    value = next(iter(preferred_values))
    if value == final_row.applicant_name:
        return final_row
    intake_values = {candidate.value for candidate in intakes}
    if intake_values and value in intake_values:
        return final_row
    if intakes:
        best_pref = max(c.ocr_confidence for c in preferred)
        best_intake = max(c.ocr_confidence for c in intakes)
        if best_pref < best_intake + 0.05:
            return final_row

    payload = final_row.to_dict()
    payload["applicant_name"] = value
    return PredictionRow.from_mapping(payload, fallback_case_id=case_id)
