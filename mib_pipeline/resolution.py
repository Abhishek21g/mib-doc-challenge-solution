"""Active-case linking and deterministic evidence precedence resolution."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .extraction import CandidateEvidence, EvidenceType
from .models import CASE_ID_PATTERN, FIELD_NAMES


POLICY_ONLY_FIELDS = (
    "stay_duration_days",
    "packet_receipt_date",
    "biohazard_check",
    "hardship_waiver",
    "diplomatic_waiver_code",
    "diplomatic_note",
    "minimal_diplomatic_packet",
    "work_permit_requested",
    "page_type_present_fee_receipt",
    "page_type_present_other",
    "page_type_present_sponsor_attestation",
)

# These values describe the case packet rather than establishing identity.
# In a visibly multi-applicant packet, a mildly different OCR name can scope a
# lower-precedence page away even though that page contains the *only* visible
# value for one of these fields.  Such a value is safe to retain only when the
# whole expected-case evidence set agrees on it.  Identity, risk, decisions,
# and policy-only facts deliberately remain subject to ordinary applicant
# scoping because a wrong association would change the safety outcome.
UNIQUE_CASE_FIELD_FALLBACKS = frozenset(
    {
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "fee_status",
    }
)
RESOLVABLE_FIELDS = (
    tuple(field for field in FIELD_NAMES if field != "confidence")
    + POLICY_ONLY_FIELDS
)


class FieldState(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"
    CONTESTED = "contested"


@dataclass(frozen=True)
class LinkedCase:
    case_id: str
    active_applicant: str | None
    evidence: tuple[CandidateEvidence, ...]
    unresolved: bool
    unresolved_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedField:
    field_name: str
    state: FieldState
    value: str | None
    winning_evidence: CandidateEvidence | None
    considered: tuple[CandidateEvidence, ...]
    reason: str


@dataclass(frozen=True)
class ResolvedCase:
    case_id: str
    active_applicant: str | None
    fields: Mapping[str, ResolvedField]
    unresolved_linkage: bool
    unresolved_reasons: tuple[str, ...]
    rescinded_decision: bool = False

    def value(self, field_name: str) -> str | None:
        field = self.fields[field_name]
        return field.value if field.state is FieldState.RESOLVED else None

    @property
    def contested_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name, field in self.fields.items()
            if field.state is FieldState.CONTESTED
        )

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name, field in self.fields.items()
            if field.state is FieldState.UNKNOWN
        )


class EvidencePrecedenceHierarchy:
    """Binding six-level hierarchy from the MIB field manual."""

    _RANKS = {
        EvidenceType.ADJUDICATOR_STAMP: 1,
        EvidenceType.SIGNED_MANUAL_NOTE: 1,
        EvidenceType.INTAKE_FORM: 2,
        EvidenceType.BIOMETRIC_SLIP: 3,
        EvidenceType.SPONSOR_ATTESTATION: 4,
        EvidenceType.REGISTRY_EXTRACT: 5,
        EvidenceType.TEXT_LAYER: 6,
    }

    @classmethod
    def rank(cls, evidence_type: EvidenceType) -> int:
        return cls._RANKS[evidence_type]


class CaseLinker:
    """Scope evidence to the case filename and its reliably-linked applicant."""

    _SPONSOR_CORROBORATION_SIMILARITY = 0.65
    _SPONSOR_MINIMUM_CONFIDENCE = 0.90
    _SUPPORT_MINIMUM_CONFIDENCE = 0.75
    _LOW_INTAKE_CONFIDENCE = 0.72
    _CONFLICTING_NAME_SIMILARITY = 0.90

    @staticmethod
    def _unique_case_field_evidence(
        candidates: Iterable[CandidateEvidence],
    ) -> set[int]:
        """Return identities of unambiguous case-level structured evidence.

        This is intentionally an inclusion fallback, not a precedence rule:
        the resolver still applies the published evidence hierarchy after the
        candidates are retained.  Any visible disagreement, supersession, or
        strike keeps the field fully applicant-scoped.
        """

        candidates = tuple(candidates)
        retained: set[int] = set()
        for field_name in UNIQUE_CASE_FIELD_FALLBACKS:
            eligible = tuple(
                candidate
                for candidate in candidates
                if candidate.field_name == field_name
                and candidate.legible
                and candidate.value is not None
                and not candidate.superseded
                and "strikethrough" not in candidate.visual_cues
                and "sample_denial_watermark" not in candidate.visual_cues
            )
            if len({candidate.value for candidate in eligible}) == 1:
                retained.update(id(candidate) for candidate in eligible)
        return retained

    @staticmethod
    def _name_similarity(left: str, right: str) -> float:
        left_key = re.sub(r"[^a-z0-9]+", "", left.casefold())
        right_key = re.sub(r"[^a-z0-9]+", "", right.casefold())
        if not left_key or not right_key:
            return 0.0
        return difflib.SequenceMatcher(None, left_key, right_key).ratio()

    @classmethod
    def _corroborated_sponsor_applicant(
        cls,
        applicant_evidence: Iterable[CandidateEvidence],
    ) -> tuple[str, set[str]] | None:
        """Prefer a structured sponsor name only over a damaged intake read.

        The override is deliberately narrow: a high-confidence applicant from
        the repeated sponsor sentence must be independently corroborated by a
        registry or biometric applicant. Every conflicting intake applicant
        must be below the refinement confidence gate. The returned aliases omit
        those damaged intake readings so their page-scoped values cannot regain
        precedence over the corroborated sources.
        """

        applicant_evidence = tuple(applicant_evidence)
        structured_sponsors = tuple(
            candidate
            for candidate in applicant_evidence
            if candidate.evidence_type is EvidenceType.SPONSOR_ATTESTATION
            and "structured_sponsor_narrative" in candidate.visual_cues
            and candidate.value is not None
            and candidate.ocr_confidence >= cls._SPONSOR_MINIMUM_CONFIDENCE
        )
        sponsor_values = {
            candidate.value for candidate in structured_sponsors if candidate.value
        }
        if len(sponsor_values) != 1:
            return None
        sponsor_value = next(iter(sponsor_values))
        supporting = tuple(
            candidate
            for candidate in applicant_evidence
            if candidate.evidence_type
            in {EvidenceType.BIOMETRIC_SLIP, EvidenceType.REGISTRY_EXTRACT}
            and candidate.value is not None
            and candidate.ocr_confidence >= cls._SUPPORT_MINIMUM_CONFIDENCE
            and cls._name_similarity(sponsor_value, candidate.value)
            >= cls._SPONSOR_CORROBORATION_SIMILARITY
        )
        if not supporting:
            return None
        conflicting_intake = tuple(
            candidate
            for candidate in applicant_evidence
            if candidate.evidence_type is EvidenceType.INTAKE_FORM
            and candidate.value is not None
            and cls._name_similarity(sponsor_value, candidate.value)
            < cls._CONFLICTING_NAME_SIMILARITY
        )
        if not conflicting_intake or any(
            candidate.ocr_confidence >= cls._LOW_INTAKE_CONFIDENCE
            for candidate in conflicting_intake
        ):
            return None
        aliases = {sponsor_value}
        aliases.update(
            candidate.value for candidate in supporting if candidate.value is not None
        )
        return sponsor_value, aliases

    @staticmethod
    def _clean_scoped_candidates(
        evidence: Iterable[CandidateEvidence],
        field_name: str,
    ) -> tuple[CandidateEvidence, ...]:
        return tuple(
            candidate
            for candidate in evidence
            if candidate.field_name == field_name
            and candidate.legible
            and candidate.value is not None
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
            and "sample_denial_watermark" not in candidate.visual_cues
        )

    @staticmethod
    def _physical_source_signature(
        candidate: CandidateEvidence,
    ) -> tuple[object, ...]:
        """Identify one visible fact while ignoring its sequential name hint."""

        return (
            candidate.value,
            candidate.evidence_type,
            candidate.page_index,
            candidate.box,
            candidate.ocr_confidence,
            candidate.visual_cues,
            candidate.source,
        )

    @classmethod
    def _non_identity_scope_is_stable(
        cls,
        ordinary: tuple[CandidateEvidence, ...],
        alternative: tuple[CandidateEvidence, ...],
    ) -> bool:
        """Allow a corroborated name only when every other visible fact is stable."""

        field_names = {
            candidate.field_name
            for evidence in (ordinary, alternative)
            for candidate in evidence
        } - {"case_id", "applicant_name"}
        source_stable_fields = {
            "adjudication",
            "risk_flags",
            *POLICY_ONLY_FIELDS,
        }
        for field_name in field_names:
            old = cls._clean_scoped_candidates(ordinary, field_name)
            new = cls._clean_scoped_candidates(alternative, field_name)
            if {candidate.value for candidate in old} != {
                candidate.value for candidate in new
            }:
                return False
            if field_name in source_stable_fields and {
                cls._physical_source_signature(candidate) for candidate in old
            } != {
                cls._physical_source_signature(candidate) for candidate in new
            }:
                return False
        return True

    @staticmethod
    def _exact_lower_corroboration(
        case_id: str,
        lower_clusters: Iterable[list[CandidateEvidence]],
    ) -> tuple[str, set[str]] | None:
        """Find one verbatim lower name repeated across pages and source types."""

        by_value: dict[str, list[CandidateEvidence]] = {}
        for cluster in lower_clusters:
            for candidate in cluster:
                if candidate.value is not None:
                    by_value.setdefault(candidate.value, []).append(candidate)
        qualifying = [
            candidates
            for candidates in by_value.values()
            if len(candidates) >= 2
            and len({candidate.page_index for candidate in candidates}) >= 2
            and len({candidate.evidence_type for candidate in candidates}) >= 2
            and all(
                candidate.case_id_hint == case_id
                and candidate.legible
                and candidate.value is not None
                and not candidate.superseded
                and "strikethrough" not in candidate.visual_cues
                and "sample_denial_watermark" not in candidate.visual_cues
                for candidate in candidates
            )
        ]
        if len(qualifying) != 1:
            return None
        value = qualifying[0][0].value
        return (value, {value}) if value is not None else None

    def link(
        self,
        expected_case_id: str | None,
        candidates: Iterable[CandidateEvidence],
    ) -> LinkedCase:
        candidates = tuple(candidates)
        expected = (
            expected_case_id.strip()
            if isinstance(expected_case_id, str)
            and CASE_ID_PATTERN.fullmatch(expected_case_id.strip())
            else None
        )
        visible_case_ids = {
            candidate.value
            for candidate in candidates
            if candidate.field_name == "case_id"
            and candidate.legible
            and candidate.value is not None
            and CASE_ID_PATTERN.fullmatch(candidate.value)
        }
        reasons: list[str] = []
        if expected is not None:
            case_id = expected
            if visible_case_ids and expected not in visible_case_ids:
                reasons.append("visible case_id conflicts with source filename")
        elif len(visible_case_ids) == 1:
            case_id = next(iter(visible_case_ids))
        else:
            case_id = ""
            reasons.append("active case_id cannot be determined")

        case_scoped = tuple(
            candidate
            for candidate in candidates
            if not candidate.case_id_hint
            or not case_id
            or candidate.case_id_hint == case_id
        )
        applicant_evidence = tuple(
            candidate
            for candidate in case_scoped
            if candidate.field_name == "applicant_name"
            and candidate.legible
            and candidate.value
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
        )
        clusters: list[list[CandidateEvidence]] = []
        for candidate in applicant_evidence:
            matching_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if any(
                        self._name_similarity(candidate.value or "", item.value or "")
                        >= 0.80
                        for item in cluster
                    )
                ),
                None,
            )
            if matching_cluster is None:
                clusters.append([candidate])
            else:
                matching_cluster.append(candidate)

        def cluster_strength(cluster: list[CandidateEvidence]) -> tuple[int, int, int, float]:
            ranks = [
                EvidencePrecedenceHierarchy.rank(candidate.evidence_type)
                for candidate in cluster
            ]
            best_rank = min(ranks)
            return (
                -best_rank,
                ranks.count(best_rank),
                len(cluster),
                max(candidate.ocr_confidence for candidate in cluster),
            )

        ranked_clusters = sorted(clusters, key=cluster_strength, reverse=True)
        active_aliases: set[str] = set()
        corroborated_sponsor = self._corroborated_sponsor_applicant(
            applicant_evidence
        )
        if corroborated_sponsor is not None:
            active_applicant, active_aliases = corroborated_sponsor
        elif ranked_clusters:
            winning_cluster = ranked_clusters[0]
            tied = (
                len(ranked_clusters) > 1
                and cluster_strength(ranked_clusters[1]) == cluster_strength(winning_cluster)
            )
            if tied:
                active_applicant = None
                reasons.append("multiple applicants cannot be reliably separated")
            else:
                representative = max(
                    winning_cluster,
                    key=lambda candidate: (
                        candidate.ocr_confidence,
                        -EvidencePrecedenceHierarchy.rank(candidate.evidence_type),
                    ),
                )
                active_applicant = representative.value
                active_aliases = {
                    candidate.value
                    for candidate in winning_cluster
                    if candidate.value is not None
                }
        else:
            active_applicant = None

        def scope_evidence(
            selected_applicant: str | None,
            selected_aliases: set[str],
        ) -> tuple[CandidateEvidence, ...]:
            # A visible same-page manual applicant correction changes the
            # subject of the intake record without invalidating earlier fields.
            correction_pages = {
                candidate.page_index
                for candidate in case_scoped
                if candidate.field_name == "applicant_name"
                and candidate.value in selected_aliases
                and "correction" in candidate.visual_cues
                and candidate.legible
                and not candidate.superseded
            }
            corrected_page_aliases = {
                (candidate.page_index, candidate.value)
                for candidate in case_scoped
                if candidate.page_index in correction_pages
                and candidate.field_name == "applicant_name"
                and candidate.value
                and candidate.value not in selected_aliases
                and (
                    candidate.superseded
                    or "strikethrough" in candidate.visual_cues
                )
            }
            if selected_applicant is None and len(clusters) > 1:
                return tuple(
                    candidate
                    for candidate in case_scoped
                    if candidate.field_name in {"case_id", "applicant_name"}
                )
            unique_case_evidence = (
                self._unique_case_field_evidence(case_scoped)
                if selected_applicant is not None
                and len(clusters) > 1
                and not reasons
                else set()
            )
            return tuple(
                candidate
                for candidate in case_scoped
                if not candidate.applicant_hint
                or not selected_applicant
                or candidate.applicant_hint in selected_aliases
                or (candidate.page_index, candidate.applicant_hint)
                in corrected_page_aliases
                or id(candidate) in unique_case_evidence
            )

        applicant_scoped = scope_evidence(active_applicant, active_aliases)
        if (
            corroborated_sponsor is None
            and not reasons
            and active_applicant is not None
            and len(ranked_clusters) >= 2
            and len(ranked_clusters[0]) == 1
        ):
            current = ranked_clusters[0][0]
            lower_corroboration = self._exact_lower_corroboration(
                case_id,
                ranked_clusters[1:],
            )
            if (
                current.evidence_type is EvidenceType.INTAKE_FORM
                and current.case_id_hint == case_id
                and not current.superseded
                and "strikethrough" not in current.visual_cues
                and "sample_denial_watermark" not in current.visual_cues
                and active_applicant == current.value
                and lower_corroboration is not None
            ):
                alternative_applicant, alternative_aliases = lower_corroboration
                alternative_scoped = scope_evidence(
                    alternative_applicant,
                    alternative_aliases,
                )
                if self._non_identity_scope_is_stable(
                    applicant_scoped,
                    alternative_scoped,
                ):
                    active_applicant = alternative_applicant
                    active_aliases = alternative_aliases
                    applicant_scoped = alternative_scoped

        return LinkedCase(
            case_id=case_id,
            active_applicant=active_applicant,
            evidence=applicant_scoped,
            unresolved=bool(reasons),
            unresolved_reasons=tuple(reasons),
        )


class RescindedDecisionHandler:
    """Neutralize decorative or visibly overturned denial decisions."""

    @staticmethod
    def _sequence(candidate: CandidateEvidence) -> tuple[int, float]:
        return candidate.page_index, candidate.box.top

    def filter(
        self,
        candidates: Iterable[CandidateEvidence],
    ) -> tuple[tuple[CandidateEvidence, ...], bool]:
        candidates = tuple(candidates)
        eligible = [
            candidate
            for candidate in candidates
            if "sample_denial_watermark" not in candidate.visual_cues
        ]
        decisions = [
            candidate
            for candidate in eligible
            if candidate.field_name == "adjudication" and candidate.value
        ]
        later_signed_approvals = [
            candidate
            for candidate in decisions
            if candidate.value == "APPROVED"
            and candidate.evidence_type is EvidenceType.SIGNED_MANUAL_NOTE
            and "correction" in candidate.visual_cues
        ]
        rescinded = False
        if later_signed_approvals:
            latest_approval = max(later_signed_approvals, key=self._sequence)
            filtered: list[CandidateEvidence] = []
            for candidate in eligible:
                is_overturned_denial = (
                    candidate.field_name == "adjudication"
                    and candidate.value == "DENIED"
                    and candidate.evidence_type is EvidenceType.ADJUDICATOR_STAMP
                    and self._sequence(candidate) < self._sequence(latest_approval)
                )
                if is_overturned_denial:
                    rescinded = True
                    continue
                filtered.append(candidate)
            eligible = filtered
        return tuple(eligible), rescinded


class EvidencePrecedenceResolver:
    """Resolve one coherent field set after case/applicant scoping."""

    _STRUCTURED_SPONSOR_REPAIR_FIELDS = frozenset(
        {"sponsor_id", "visa_class"}
    )
    _STRUCTURED_SPONSOR_MINIMUM_CONFIDENCE = 0.90
    _STRUCTURED_SPONSOR_NAME_SIMILARITY = 0.80

    def __init__(
        self,
        *,
        hierarchy: type[EvidencePrecedenceHierarchy] = EvidencePrecedenceHierarchy,
        rescinded_handler: RescindedDecisionHandler | None = None,
    ) -> None:
        self._hierarchy = hierarchy
        self._rescinded = rescinded_handler or RescindedDecisionHandler()

    def _resolve_field(
        self,
        field_name: str,
        candidates: Iterable[CandidateEvidence],
    ) -> ResolvedField:
        considered = tuple(
            candidate
            for candidate in candidates
            if candidate.field_name == field_name
        )
        eligible = tuple(
            candidate
            for candidate in considered
            if candidate.legible
            and candidate.value is not None
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
        )
        if not eligible:
            return ResolvedField(
                field_name=field_name,
                state=FieldState.UNKNOWN,
                value=None,
                winning_evidence=None,
                considered=considered,
                reason="no eligible evidence",
            )

        winning_rank = min(
            self._hierarchy.rank(candidate.evidence_type) for candidate in eligible
        )
        top_rank = tuple(
            candidate
            for candidate in eligible
            if self._hierarchy.rank(candidate.evidence_type) == winning_rank
        )
        corrections = tuple(
            candidate
            for candidate in top_rank
            if "correction" in candidate.visual_cues
        )
        finalists = corrections or top_rank
        values = {candidate.value for candidate in finalists}
        if len(values) != 1:
            return ResolvedField(
                field_name=field_name,
                state=FieldState.CONTESTED,
                value=None,
                winning_evidence=None,
                considered=considered,
                reason=f"same-rank conflict at precedence rank {winning_rank}",
            )
        value = next(iter(values))
        winning_evidence = max(
            finalists,
            key=lambda candidate: (
                candidate.ocr_confidence,
                candidate.page_index,
                candidate.box.top,
            ),
        )
        return ResolvedField(
            field_name=field_name,
            state=FieldState.RESOLVED,
            value=value,
            winning_evidence=winning_evidence,
            considered=considered,
            reason=f"resolved at precedence rank {winning_rank}",
        )

    @classmethod
    def _structured_sponsor_conflict_repair(
        cls,
        field_name: str,
        case_id: str,
        candidates: Iterable[CandidateEvidence],
    ) -> ResolvedField | None:
        """Repair two narrow OCR conflicts using a redundant sponsor sentence.

        Sponsor prose repeats the applicant, sponsor ID, and visa class in one
        structured sentence.  It is usable as an OCR repair only when every
        relevant source is clean, explicitly exact-case, applicant-aligned,
        and the high-confidence sponsor reading has one unique value.  This is
        not a general sponsor-over-intake precedence override.
        """

        if field_name not in cls._STRUCTURED_SPONSOR_REPAIR_FIELDS or not case_id:
            return None
        considered = tuple(
            candidate
            for candidate in candidates
            if candidate.field_name == field_name
        )

        def clean(candidate: CandidateEvidence) -> bool:
            return (
                candidate.legible
                and candidate.value is not None
                and not candidate.superseded
                and "strikethrough" not in candidate.visual_cues
                and "sample_denial_watermark" not in candidate.visual_cues
                and candidate.case_id_hint == case_id
                and bool(candidate.applicant_hint)
            )

        intake = tuple(
            candidate
            for candidate in considered
            if candidate.evidence_type is EvidenceType.INTAKE_FORM
            and clean(candidate)
        )
        sponsor = tuple(
            candidate
            for candidate in considered
            if candidate.evidence_type is EvidenceType.SPONSOR_ATTESTATION
            and "structured_sponsor_narrative" in candidate.visual_cues
            and candidate.ocr_confidence
            >= cls._STRUCTURED_SPONSOR_MINIMUM_CONFIDENCE
            and clean(candidate)
        )
        if not intake or not sponsor or any(
            "correction" in candidate.visual_cues for candidate in intake
        ):
            return None

        intake_values = {candidate.value for candidate in intake}
        sponsor_values = {candidate.value for candidate in sponsor}
        if (
            len(intake_values) != 1
            or len(sponsor_values) != 1
            or intake_values == sponsor_values
            or any(
                CaseLinker._name_similarity(
                    intake_candidate.applicant_hint or "",
                    sponsor_candidate.applicant_hint or "",
                )
                < cls._STRUCTURED_SPONSOR_NAME_SIMILARITY
                for intake_candidate in intake
                for sponsor_candidate in sponsor
            )
        ):
            return None

        value = next(iter(sponsor_values))
        winning_evidence = max(
            sponsor,
            key=lambda candidate: (
                candidate.ocr_confidence,
                candidate.page_index,
                candidate.box.top,
            ),
        )
        return ResolvedField(
            field_name=field_name,
            state=FieldState.RESOLVED,
            value=value,
            winning_evidence=winning_evidence,
            considered=considered,
            reason="structured exact-case sponsor narrative OCR repair",
        )

    def resolve(self, linked_case: LinkedCase) -> ResolvedCase:
        evidence, rescinded = self._rescinded.filter(linked_case.evidence)
        fields = {}
        for field_name in RESOLVABLE_FIELDS:
            fields[field_name] = (
                self._structured_sponsor_conflict_repair(
                    field_name,
                    linked_case.case_id,
                    evidence,
                )
                or self._resolve_field(field_name, evidence)
            )
        if linked_case.case_id:
            case_candidates = tuple(
                candidate
                for candidate in evidence
                if candidate.field_name == "case_id"
            )
            fields["case_id"] = ResolvedField(
                field_name="case_id",
                state=FieldState.RESOLVED,
                value=linked_case.case_id,
                winning_evidence=(case_candidates[0] if case_candidates else None),
                considered=case_candidates,
                reason="active case association",
            )
        if linked_case.active_applicant:
            applicant_candidates = tuple(
                candidate
                for candidate in evidence
                if candidate.field_name == "applicant_name"
                and candidate.value == linked_case.active_applicant
            )
            fields["applicant_name"] = ResolvedField(
                field_name="applicant_name",
                state=FieldState.RESOLVED,
                value=linked_case.active_applicant,
                winning_evidence=(
                    applicant_candidates[0] if applicant_candidates else None
                ),
                considered=applicant_candidates,
                reason="active applicant association",
            )
        return ResolvedCase(
            case_id=linked_case.case_id,
            active_applicant=linked_case.active_applicant,
            fields=fields,
            unresolved_linkage=linked_case.unresolved,
            unresolved_reasons=linked_case.unresolved_reasons,
            rescinded_decision=rescinded,
        )
