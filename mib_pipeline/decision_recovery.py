"""Frozen, identity-free recovery of a narrow subset of review decisions."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Protocol

from .adjudication import AdjudicationOutcome
from .extraction import EvidenceType
from .models import PredictionRow
from .resolution import FieldState, ResolvedCase


REVIEW_DENIAL_CONFIDENCE = 0.551819438046983
REVIEW_APPROVAL_CONFIDENCE = 0.98
# Kept as a public compatibility alias for the first approval recovery head.
REVIEW_DIPLOMATIC_APPROVAL_CONFIDENCE = REVIEW_APPROVAL_CONFIDENCE
_SNAPSHOT_DATE = date(2026, 7, 7)
_MAXIMUM_BASELINE_CONFIDENCE = 0.35
_MAXIMUM_DIPLOMATIC_APPROVAL_BASELINE_CONFIDENCE = 0.25
_MAXIMUM_SPONSOR_XW1_APPROVAL_BASELINE_CONFIDENCE = 0.20
_MINIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE = 0.20
_MAXIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE = 0.25
_FEE_RECEIPT_MARKER = "page_type_present_fee_receipt"
_OTHER_MARKER = "page_type_present_other"
_SPONSOR_ATTESTATION_MARKER = "page_type_present_sponsor_attestation"
_MARKER_CUES = {
    _FEE_RECEIPT_MARKER: "packet_page_type:fee_receipt",
    _OTHER_MARKER: "packet_page_type:other",
    _SPONSOR_ATTESTATION_MARKER: "packet_page_type:sponsor_attestation",
}


class OutcomeAdjudicator(Protocol):
    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        """Return the baseline policy outcome and trace."""


class ReviewDenialRecoveryAdjudicator:
    """Apply frozen identity-free recovery rules to baseline reviews.

    The wrapper consumes only generic policy trace categories, bounded output
    categories (arrival age and visa class), the baseline confidence, and
    packet-level visible page-type markers.  Case IDs, names, filenames, and
    truth labels are never inspected.  All row values other than
    adjudication/confidence remain byte-for-byte equivalent at the typed model
    boundary.
    """

    def __init__(self, baseline: OutcomeAdjudicator) -> None:
        self._baseline = baseline

    @staticmethod
    def _visible_marker(resolved_case: ResolvedCase, field_name: str) -> bool:
        field = resolved_case.fields.get(field_name)
        evidence = field.winning_evidence if field is not None else None
        return bool(
            field is not None
            and field.state is FieldState.RESOLVED
            and field.value == "present"
            and evidence is not None
            and evidence.legible
            and not evidence.superseded
            and evidence.source == "visible_ocr"
            and evidence.evidence_type is not EvidenceType.TEXT_LAYER
            and "strikethrough" not in evidence.visual_cues
            and "sample_denial_watermark" not in evidence.visual_cues
            and _MARKER_CUES.get(field_name) in evidence.visual_cues
        )

    @classmethod
    def _visible_marker_page_count(
        cls,
        resolved_case: ResolvedCase,
        field_name: str,
    ) -> int:
        field = resolved_case.fields.get(field_name)
        if not cls._visible_marker(resolved_case, field_name) or field is None:
            return 0
        cue = _MARKER_CUES.get(field_name)
        pages = {
            evidence.page_index
            for evidence in field.considered
            if evidence.value == "present"
            and evidence.legible
            and not evidence.superseded
            and evidence.source == "visible_ocr"
            and evidence.evidence_type is not EvidenceType.TEXT_LAYER
            and "strikethrough" not in evidence.visual_cues
            and "sample_denial_watermark" not in evidence.visual_cues
            and cue in evidence.visual_cues
        }
        return len(pages)

    @staticmethod
    def _arrival_is_stale_gt365(row: PredictionRow) -> bool:
        # This schema fallback means "unresolved"; it is not visible evidence
        # of an old arrival and must never trigger a pre-Rapid denial.
        if row.arrival_date == "1900-01-01":
            return False
        try:
            arrival = date.fromisoformat(row.arrival_date)
        except ValueError:
            return False
        return (_SNAPSHOT_DATE - arrival).days > 365

    @staticmethod
    def _visible_diplomatic_visa(resolved_case: ResolvedCase) -> bool:
        field = resolved_case.fields.get("visa_class")
        evidence = field.winning_evidence if field is not None else None
        return bool(
            field is not None
            and field.state is FieldState.RESOLVED
            and field.value == "DIP-1"
            and evidence is not None
            and evidence.legible
            and not evidence.superseded
            and evidence.source == "visible_ocr"
            and evidence.evidence_type is not EvidenceType.TEXT_LAYER
            and "strikethrough" not in evidence.visual_cues
            and "sample_denial_watermark" not in evidence.visual_cues
            and evidence.case_id_hint in {None, resolved_case.case_id}
            and evidence.applicant_hint
            in {None, resolved_case.active_applicant}
        )

    @classmethod
    def _matches_diplomatic_fee_approval(
        cls,
        resolved_case: ResolvedCase,
        outcome: AdjudicationOutcome,
    ) -> bool:
        if outcome.row.confidence > _MAXIMUM_DIPLOMATIC_APPROVAL_BASELINE_CONFIDENCE:
            return False
        if outcome.trace.denial_reasons:
            return False
        diplomatic_fact = (
            "diplomatic_sponsor_exemption" in outcome.trace.approval_facts
        )
        return (
            cls._visible_marker_page_count(
                resolved_case,
                _FEE_RECEIPT_MARKER,
            )
            == 1
            and (diplomatic_fact or cls._visible_diplomatic_visa(resolved_case))
        )

    @classmethod
    def _matching_rules(
        cls,
        resolved_case: ResolvedCase,
        outcome: AdjudicationOutcome,
    ) -> tuple[str, ...]:
        reasons = frozenset(outcome.trace.review_reasons)
        low_confidence = outcome.row.confidence <= _MAXIMUM_BASELINE_CONFIDENCE
        matches: list[str] = []
        if (
            low_confidence
            and "clean_biohazard_check_missing" in reasons
            and cls._visible_marker(resolved_case, _OTHER_MARKER)
        ):
            matches.append("review_denial_other_missing_biohazard")
        if (
            low_confidence
            and cls._arrival_is_stale_gt365(outcome.row)
            and cls._visible_marker(resolved_case, _SPONSOR_ATTESTATION_MARKER)
        ):
            matches.append("review_denial_sponsor_stale_gt365")
        if {
            "required_output_unknown:home_world",
            "required_output_unknown:risk_flags",
            "required_output_unknown:sponsor_id",
        }.issubset(reasons):
            matches.append("review_denial_three_required_outputs_unknown")
        return tuple(matches)

    @classmethod
    def _matching_approval_rules(
        cls,
        resolved_case: ResolvedCase,
        outcome: AdjudicationOutcome,
    ) -> tuple[str, ...]:
        """Return frozen approval heads in stable evaluation order."""

        # An approval recovery must never erase an explicit policy denial,
        # including one supplied by a malformed/custom baseline.
        if outcome.trace.denial_reasons:
            return ()

        confidence = outcome.row.confidence
        reasons = frozenset(outcome.trace.review_reasons)
        facts = frozenset(outcome.trace.approval_facts)
        current_application = "application_date_current_or_exempt" in facts
        matches: list[str] = []

        if cls._matches_diplomatic_fee_approval(resolved_case, outcome):
            matches.append("review_diplomatic_fee_receipt_recovery")
        if (
            confidence <= _MAXIMUM_SPONSOR_XW1_APPROVAL_BASELINE_CONFIDENCE
            and outcome.row.visa_class == "XW-1"
            and cls._visible_marker(
                resolved_case,
                _SPONSOR_ATTESTATION_MARKER,
            )
        ):
            matches.append("review_approval_sponsor_attestation_xw1")
        if (
            _MINIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE
            < confidence
            <= _MAXIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE
            and current_application
            and "visa_class_unknown" in reasons
        ):
            matches.append("review_approval_current_application_visa_unknown")
        if (
            confidence <= _MAXIMUM_BASELINE_CONFIDENCE
            and current_application
            and "required_output_unknown:home_world" in reasons
            and "unsupported_fee_waiver" not in reasons
        ):
            matches.append("review_approval_current_application_home_world_unknown")
        return tuple(matches)

    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        baseline = self._baseline.adjudicate_case(resolved_case)
        # Requiring both typed outputs to agree makes a malformed/custom
        # baseline fail closed instead of broadening the override surface.
        if (
            baseline.row.adjudication != "NEEDS_REVIEW"
            or baseline.trace.decision != "NEEDS_REVIEW"
        ):
            return baseline
        matching_rules = self._matching_rules(resolved_case, baseline)
        if matching_rules:
            trace = replace(
                baseline.trace,
                decision="DENIED",
                authoritative_source=False,
                denial_reasons=tuple(
                    sorted(set(baseline.trace.denial_reasons) | set(matching_rules))
                ),
            )
            row = replace(
                baseline.row,
                adjudication="DENIED",
                confidence=REVIEW_DENIAL_CONFIDENCE,
            )
            return AdjudicationOutcome(row=row, trace=trace)

        matching_approval_rules = self._matching_approval_rules(
            resolved_case,
            baseline,
        )
        if not matching_approval_rules:
            return baseline
        trace = replace(
            baseline.trace,
            decision="APPROVED",
            authoritative_source=False,
            review_reasons=(),
            approval_facts=tuple(
                sorted(
                    set(baseline.trace.approval_facts)
                    | set(matching_approval_rules)
                )
            ),
        )
        row = replace(
            baseline.row,
            adjudication="APPROVED",
            confidence=REVIEW_APPROVAL_CONFIDENCE,
        )
        return AdjudicationOutcome(row=row, trace=trace)

    def adjudicate(self, resolved_case: ResolvedCase) -> PredictionRow:
        return self.adjudicate_case(resolved_case).row
