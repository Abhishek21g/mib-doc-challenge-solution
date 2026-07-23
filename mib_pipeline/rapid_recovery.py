"""Fail-closed RapidOCR recovery for genuinely unresolved output fields.

The primary Tesseract pass remains the source of case scope, policy, and
confidence.  RapidOCR is a second, independently resolved reading of the same
rendered pixels.  It may fill only primary ``FieldState.UNKNOWN`` output
values.  One frozen semantic head may turn a remaining review into a denial
when those newly visible values establish a published disqualifier; it never
creates an approval.  One separate frozen tie-breaker may repair only the
serialized applicant name when the already-extracted primary evidence
contains one stronger, exact-case biometric value.
"""

from __future__ import annotations

import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .adjudication import AdjudicationOutcome, PolicyRuleSet
from .extraction import (
    CandidateEvidence,
    EvidenceType,
    OcrToken,
    VisibleEvidenceExtractor,
)
from .ingestion import Rect, RenderedCase, RenderedPage
from .arjun_heads import apply_visible_field_repairs
from .models import PredictionRow
from .resolution import FieldState, ResolvedCase, ResolvedField


# These are the scored output values for which the frozen liberal recovery was
# measured.  Case identity and adjudication are deliberately absent.  Risk is
# handled by its narrower, non-``none`` condition below.
RAPID_OUTPUT_FIELDS = frozenset(
    {
        "applicant_name",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "fee_status",
    }
)
RAPID_RISK_FIELD = "risk_flags"
RAPID_RISK_ROUTE_FIELDS = frozenset(
    {
        "species_code",
        "home_world",
        "visa_class",
        "arrival_date",
        "declared_purpose",
        "fee_status",
    }
)
AUTHORITATIVE_RAPID_TYPES = frozenset(
    {EvidenceType.ADJUDICATOR_STAMP, EvidenceType.SIGNED_MANUAL_NOTE}
)
AUTHORITATIVE_MINIMUM_CONFIDENCE = 0.90
BIOMETRIC_APPLICANT_MINIMUM_CONFIDENCE = 0.80
SOURCE_PRIORITY_MINIMUM_CONFIDENCE = 0.90
RAPID_BAD_CUES = frozenset({"strikethrough", "sample_denial_watermark"})
SEMANTIC_DENIAL_CONFIDENCE = 0.9166666666666666
REVIEW_APPROVAL_CONFIDENCE = 0.80
XW1_MULTISOURCE_REVIEW_APPROVAL_CONFIDENCE = 0.98
XW1_MULTISOURCE_COMPLETE_REVIEW_RECOVERY = (
    "xw1_multisource_complete_review_recovery"
)
_COMPLETE_REVIEW_OUTPUT_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)
_INCOMPLETE_REVIEW_VALUES = frozenset({"", "unknown", "null"})
_REVIEW_APPROVAL_SNAPSHOT_DATE = date(2026, 7, 7)
SEMANTIC_POLICY_RULES = PolicyRuleSet()
SEMANTIC_EVIDENCE_FIELDS = frozenset(
    {"risk_flags", "home_world", "visa_class", "sponsor_id"}
)
SEMANTIC_DENIAL_RULE_IDS = (
    "semantic_disqualifying_risk",
    "semantic_absolute_embargo",
    "semantic_wolf_non_diplomatic",
    "semantic_rapid_barred_sponsor",
)


class PrimaryAdjudicator(Protocol):
    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        """Return the primary policy row and its decision trace."""


class RapidOcrEngine:
    """Adapt the wheel-bundled RapidOCR models to ``VisibleEvidenceExtractor``.

    Import and model construction are lazy: importing the MIB package or
    running fake-based unit tests does not require RapidOCR to be installed.
    Production instances use a primitive string model root because the pinned
    OmegaConf 2.0.0 cannot store ``Path`` objects.  All three models are read
    from the installed wheel; no URL or user cache is consulted.
    """

    def __init__(
        self,
        *,
        engine_factory: Callable[..., Any] | None = None,
        package_root: Path | str | None = None,
    ) -> None:
        if engine_factory is None:
            try:
                import rapidocr as rapidocr_package
            except ImportError as exc:  # pragma: no cover - production image path
                raise RuntimeError("RapidOCR is not installed") from exc
            engine_factory = rapidocr_package.RapidOCR
            package_file = getattr(rapidocr_package, "__file__", None)
            if package_file is None:
                raise RuntimeError("RapidOCR package location is unavailable")
            package_root = Path(package_file).resolve().parent
        if package_root is None:
            raise ValueError("package_root is required with a custom engine_factory")

        model_root = str(Path(package_root).resolve() / "models")
        self._engine = engine_factory(
            params={
                "Global.model_root_dir": model_root,
                "Global.log_level": "error",
                "Global.text_score": 0.30,
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )

    def read_page(self, page: RenderedPage) -> tuple[OcrToken, ...]:
        result = self._engine(page.image_png)
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if boxes is None or texts is None or scores is None:
            return ()

        tokens: list[OcrToken] = []
        for index, (box, text, score) in enumerate(
            zip(boxes, texts, scores),
            start=1,
        ):
            rendered = str(text).strip()
            if not rendered:
                continue
            points = tuple(box)
            if not points:
                continue
            xs = tuple(float(point[0]) for point in points)
            ys = tuple(float(point[1]) for point in points)
            tokens.append(
                OcrToken(
                    page_index=page.index,
                    text=rendered,
                    confidence=max(0.0, min(1.0, float(score))),
                    box=Rect(min(xs), min(ys), max(xs), max(ys)),
                    block_num=index,
                    paragraph_num=1,
                    line_num=1,
                    word_num=1,
                )
            )
        return tuple(tokens)


def build_rapid_extractor() -> VisibleEvidenceExtractor:
    """Create the exact full-page RapidOCR extractor used by the frozen run."""

    return VisibleEvidenceExtractor(
        ocr_engine=RapidOcrEngine(),
        psm6_refinement=False,
        consensus_retry=False,
        fee_receipt_retry=False,
        sparse_intake_retry=False,
        orientation_retry=False,
        trusted_scope_repair=False,
        risk_flag_retry=False,
    )


class RapidOutputRecoveryProcessor:
    """Run primary OCR once, then conservatively repair serialized output.

    A separate RapidOCR extractor is initialized lazily per worker thread.
    This avoids sharing ONNX Runtime sessions across the four-worker batch
    pool.  Any RapidOCR import, initialization, extraction, linking,
    resolution, or overlay failure returns the primary-only repaired row.
    """

    def __init__(
        self,
        *,
        renderer: Any,
        primary_extractor: Any,
        linker: Any,
        resolver: Any,
        adjudicator: PrimaryAdjudicator,
        rapid_extractor_factory: Callable[[], Any] = build_rapid_extractor,
    ) -> None:
        self._renderer = renderer
        self._primary_extractor = primary_extractor
        self._linker = linker
        self._resolver = resolver
        self._adjudicator = adjudicator
        self._rapid_extractor_factory = rapid_extractor_factory
        self._local = threading.local()

    def _rapid_extractor(self) -> Any:
        extractor = getattr(self._local, "rapid_extractor", None)
        if extractor is None:
            extractor = self._rapid_extractor_factory()
            self._local.rapid_extractor = extractor
        return extractor

    @staticmethod
    def _unknown_output_fields(resolved: ResolvedCase) -> frozenset[str]:
        return frozenset(
            field_name
            for field_name in RAPID_OUTPUT_FIELDS
            if (field := resolved.fields.get(field_name)) is not None
            and field.state is FieldState.UNKNOWN
        )

    @staticmethod
    def _risk_gap_is_safe(field: ResolvedField | None) -> bool:
        """Reproduce the frozen non-``none`` risk-routing safety condition."""

        if field is None or field.state is not FieldState.UNKNOWN:
            return False
        return all(
            candidate.value is None
            and not candidate.superseded
            and not (RAPID_BAD_CUES & set(candidate.visual_cues))
            for candidate in field.considered
        )

    @classmethod
    def _recover_non_none_risk(
        cls,
        resolved: ResolvedCase,
        unknown_fields: Iterable[str],
    ) -> bool:
        """Route the risk field only under the frozen liberal experiment gate.

        A safe unresolved primary risk needs either its own visible primary
        anchor or a separately unresolved non-identity field that already
        justifies the full-page Rapid pass.  Rapid's literal ``none`` never
        replaces the primary output; only a resolved non-``none`` value can.
        """

        risk_field = resolved.fields.get(RAPID_RISK_FIELD)
        if not cls._risk_gap_is_safe(risk_field):
            return False
        return bool(
            risk_field is not None
            and (
                risk_field.considered
                or RAPID_RISK_ROUTE_FIELDS.intersection(unknown_fields)
            )
        )

    @staticmethod
    def _authoritative_rapid_decision(
        *,
        case_id: str,
        primary_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        rapid_candidates: Iterable[CandidateEvidence],
    ) -> str | None:
        """Return one unanimous exact-case signed decision, otherwise abstain."""

        active_applicant = primary_resolved.active_applicant
        if (
            primary_outcome.row.adjudication != "NEEDS_REVIEW"
            or active_applicant is None
            or "authoritative_visible_decision"
            in primary_outcome.trace.review_reasons
        ):
            return None
        eligible = tuple(
            candidate
            for candidate in rapid_candidates
            if candidate.field_name == "adjudication"
            and candidate.value in {"APPROVED", "DENIED"}
            and candidate.evidence_type in AUTHORITATIVE_RAPID_TYPES
            and candidate.legible
            and not candidate.superseded
            and candidate.ocr_confidence >= AUTHORITATIVE_MINIMUM_CONFIDENCE
            and candidate.source == "visible_ocr"
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in {None, active_applicant}
            and not (RAPID_BAD_CUES & set(candidate.visual_cues))
        )
        decisions = {candidate.value for candidate in eligible}
        return next(iter(decisions)) if len(decisions) == 1 else None

    @staticmethod
    def _primary_authoritative_decision(outcome: AdjudicationOutcome) -> bool:
        """Treat any primary authoritative trace as a semantic-head veto."""

        marker = "authoritative_visible_decision"
        return bool(
            outcome.trace.authoritative_source
            or marker in outcome.trace.denial_reasons
            or marker in outcome.trace.review_reasons
            or marker in outcome.trace.approval_facts
        )

    @staticmethod
    def _has_authoritative_rapid_decision(
        *,
        case_id: str,
        primary_resolved: ResolvedCase,
        rapid_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence],
    ) -> bool:
        """Veto the semantic head on any exact-case Rapid authority.

        Unlike the narrow authoritative output override, the veto includes an
        explicit ``NEEDS_REVIEW``.  Abstaining is safer than allowing a
        lower-precedence policy fact to replace a visible signed decision.
        """

        applicant_aliases = {
            None,
            primary_resolved.active_applicant,
            rapid_resolved.active_applicant,
        }
        return any(
            isinstance(candidate, CandidateEvidence)
            and candidate.field_name == "adjudication"
            and candidate.value in {"APPROVED", "DENIED", "NEEDS_REVIEW"}
            and candidate.evidence_type in AUTHORITATIVE_RAPID_TYPES
            and candidate.legible
            and not candidate.superseded
            and candidate.ocr_confidence >= AUTHORITATIVE_MINIMUM_CONFIDENCE
            and candidate.source == "visible_ocr"
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in applicant_aliases
            and not (RAPID_BAD_CUES & set(candidate.visual_cues))
            for candidate in rapid_candidates
        )

    @staticmethod
    def _unsafe_pages(
        candidates: Iterable[CandidateEvidence],
    ) -> frozenset[int]:
        """Return pages carrying a visible strikethrough or sample watermark."""

        return frozenset(
            candidate.page_index
            for candidate in candidates
            if isinstance(candidate, CandidateEvidence)
            and RAPID_BAD_CUES.intersection(candidate.visual_cues)
        )

    @staticmethod
    def _visible_resolved_value(
        *,
        case_id: str,
        resolved: ResolvedCase,
        field_name: str,
        unsafe_pages: frozenset[int],
    ) -> str | None:
        """Return one exact-case visible winner, never a serialization prior."""

        resolved_field = resolved.fields.get(field_name)
        if (
            resolved_field is None
            or resolved_field.state is not FieldState.RESOLVED
            or resolved_field.value is None
        ):
            return None
        evidence = resolved_field.winning_evidence
        if (
            evidence is None
            or evidence.field_name != field_name
            or evidence.value != resolved_field.value
            or not evidence.legible
            or evidence.superseded
            or evidence.source != "visible_ocr"
            or evidence.evidence_type is EvidenceType.TEXT_LAYER
            or evidence.case_id_hint != case_id
            or evidence.applicant_hint not in {None, resolved.active_applicant}
            or evidence.page_index in unsafe_pages
            or RAPID_BAD_CUES.intersection(evidence.visual_cues)
        ):
            return None
        return resolved_field.value

    @staticmethod
    def _parse_risk_flags(value: str | None) -> frozenset[str]:
        if value in {None, "", "none"}:
            return frozenset()
        return frozenset(
            item.strip()
            for item in value.split("|")
            if item.strip() and item.strip() != "none"
        )

    @classmethod
    def _semantic_denial_rules(
        cls,
        *,
        case_id: str,
        payload: dict[str, object],
        primary_candidates: Iterable[CandidateEvidence],
        rapid_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
        rapid_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        unknown_fields: frozenset[str],
        recover_risk: bool,
    ) -> tuple[str, ...]:
        """Match the frozen, identity-free post-Rapid denial head.

        Primary and Rapid values stay separate so the barred-sponsor rule can
        require two independently recovered Rapid winners. Values used only
        for JSON schema completion never enter this method as evidence.
        """

        if (
            primary_outcome.row.adjudication != "NEEDS_REVIEW"
            or primary_outcome.trace.decision != "NEEDS_REVIEW"
            or cls._primary_authoritative_decision(primary_outcome)
            or cls._has_authoritative_rapid_decision(
                case_id=case_id,
                primary_resolved=primary_resolved,
                rapid_resolved=rapid_resolved,
                rapid_candidates=rapid_candidates,
            )
        ):
            return ()

        primary_unsafe_pages = cls._unsafe_pages(primary_candidates)
        rapid_unsafe_pages = cls._unsafe_pages(rapid_candidates)
        primary_values = {
            field_name: value
            for field_name in SEMANTIC_EVIDENCE_FIELDS
            if (
                value := cls._visible_resolved_value(
                    case_id=case_id,
                    resolved=primary_resolved,
                    field_name=field_name,
                    unsafe_pages=primary_unsafe_pages,
                )
            )
            is not None
            and payload.get(field_name) == value
        }

        rapid_values: dict[str, str] = {}
        for field_name in SEMANTIC_EVIDENCE_FIELDS & unknown_fields:
            value = cls._visible_resolved_value(
                case_id=case_id,
                resolved=rapid_resolved,
                field_name=field_name,
                unsafe_pages=rapid_unsafe_pages,
            )
            if value is not None and payload.get(field_name) == value:
                rapid_values[field_name] = value
        if recover_risk:
            rapid_risk = cls._visible_resolved_value(
                case_id=case_id,
                resolved=rapid_resolved,
                field_name=RAPID_RISK_FIELD,
                unsafe_pages=rapid_unsafe_pages,
            )
            if rapid_risk not in {None, "none"} and payload.get(
                RAPID_RISK_FIELD
            ) == rapid_risk:
                rapid_values[RAPID_RISK_FIELD] = rapid_risk

        values = {**primary_values, **rapid_values}
        matches: list[str] = []
        if cls._parse_risk_flags(values.get("risk_flags")) & (
            SEMANTIC_POLICY_RULES.disqualifying_flags
        ):
            matches.append(SEMANTIC_DENIAL_RULE_IDS[0])
        if values.get("home_world") in SEMANTIC_POLICY_RULES.embargoed_worlds:
            matches.append(SEMANTIC_DENIAL_RULE_IDS[1])
        if (
            values.get("home_world")
            in SEMANTIC_POLICY_RULES.non_diplomatic_embargoed_worlds
            and values.get("visa_class") not in {None, "DIP-1"}
        ):
            matches.append(SEMANTIC_DENIAL_RULE_IDS[2])
        if (
            rapid_values.get("sponsor_id") in SEMANTIC_POLICY_RULES.barred_sponsors
            and rapid_values.get("visa_class") not in {None, "DIP-1"}
        ):
            matches.append(SEMANTIC_DENIAL_RULE_IDS[3])
        return tuple(matches)

    @staticmethod
    def _rapid_value(resolved: ResolvedCase, field_name: str) -> str | None:
        field = resolved.fields.get(field_name)
        if field is None or field.state is not FieldState.RESOLVED:
            return None
        return field.value

    @staticmethod
    def _review_approval_arrival_age(value: str) -> int | None:
        """Return the frozen snapshot age for one exact ISO arrival date."""

        try:
            normalized = value.strip()
            if normalized == "1900-01-01":
                return None
            arrival = date.fromisoformat(normalized)
        except (AttributeError, TypeError, ValueError):
            return None
        return (_REVIEW_APPROVAL_SNAPSHOT_DATE - arrival).days

    @classmethod
    def _review_approval_head(
        cls,
        *,
        final_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_outcome: AdjudicationOutcome,
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence] = (),
        rapid_resolved: ResolvedCase | None = None,
    ) -> PredictionRow:
        """Apply the frozen identity-free three-branch review approval head.

        Existing primary or Rapid authority always vetoes this lower-precedence
        statistical recovery.  Candidate values and identities are never read:
        the first branch uses only the count of primary applicant candidates.
        """

        if (
            final_row.adjudication != "NEEDS_REVIEW"
            or " ".join(final_row.risk_flags.strip().split()).casefold()
            != "none"
            or cls._primary_authoritative_decision(primary_outcome)
            or (
                rapid_resolved is not None
                and cls._has_authoritative_rapid_decision(
                    case_id=primary_resolved.case_id,
                    primary_resolved=primary_resolved,
                    rapid_resolved=rapid_resolved,
                    rapid_candidates=rapid_candidates,
                )
            )
        ):
            return final_row

        applicant_candidate_count = sum(
            isinstance(candidate, CandidateEvidence)
            and candidate.field_name == "applicant_name"
            for candidate in primary_candidates
        )
        arrival_age = cls._review_approval_arrival_age(
            final_row.arrival_date
        )
        trace = primary_outcome.trace
        matches = bool(
            applicant_candidate_count > 5
            or (
                arrival_age is not None
                and arrival_age > 71
                and "no_visible_biohazard_risk"
                in trace.approval_facts
            )
            or (
                arrival_age is not None
                and arrival_age <= 48
                and "required_sponsor_unknown" in trace.review_reasons
            )
        )
        if not matches:
            return final_row

        payload = final_row.to_dict()
        payload["adjudication"] = "APPROVED"
        payload["confidence"] = REVIEW_APPROVAL_CONFIDENCE
        return PredictionRow.from_mapping(
            payload,
            fallback_case_id=final_row.case_id,
        )

    @staticmethod
    def _clean_multisource_candidate(candidate: object) -> bool:
        """Accept only live, legible facts read from rendered pixels."""

        return bool(
            isinstance(candidate, CandidateEvidence)
            and candidate.value is not None
            and candidate.legible
            and not candidate.superseded
            and candidate.source == "visible_ocr"
            and candidate.evidence_type is not EvidenceType.TEXT_LAYER
            and not RAPID_BAD_CUES.intersection(candidate.visual_cues)
        )

    @staticmethod
    def _complete_review_output(final_row: PredictionRow) -> bool:
        """Reject every schema fallback or substantively unknown output."""

        for field_name in _COMPLETE_REVIEW_OUTPUT_FIELDS:
            value = getattr(final_row, field_name)
            normalized = " ".join(value.strip().split()).casefold()
            if normalized in _INCOMPLETE_REVIEW_VALUES:
                return False
        return bool(
            final_row.sponsor_id != "SPN-0000"
            and final_row.arrival_date != "1900-01-01"
        )

    @classmethod
    def _same_page_source_fact(
        cls,
        *,
        case_id: str,
        active_applicant: str,
        field_name: str,
        expected_value: str,
        evidence_type: EvidenceType,
        candidates: tuple[CandidateEvidence, ...],
    ) -> bool:
        """Require exact-case facts with an exact same-page name anchor."""

        facts = tuple(
            candidate
            for candidate in candidates
            if cls._clean_multisource_candidate(candidate)
            and candidate.field_name == field_name
            and candidate.value == expected_value
            and candidate.evidence_type is evidence_type
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in {None, active_applicant}
        )
        if not facts:
            return False
        anchored_pages = {
            candidate.page_index
            for candidate in candidates
            if cls._clean_multisource_candidate(candidate)
            and candidate.field_name == "applicant_name"
            and candidate.value == active_applicant
            and candidate.evidence_type is evidence_type
            and candidate.case_id_hint == case_id
            and candidate.applicant_hint in {None, active_applicant}
        }
        return all(candidate.page_index in anchored_pages for candidate in facts)

    @classmethod
    def _multisource_conflict(
        cls,
        *,
        case_id: str,
        active_applicant: str,
        expected: tuple[tuple[str, str, EvidenceType], ...],
        candidates: tuple[CandidateEvidence, ...],
    ) -> bool:
        """Veto any relevant live source disagreement or applicant mismatch."""

        for field_name, expected_value, evidence_type in expected:
            if any(
                cls._clean_multisource_candidate(candidate)
                and candidate.field_name == field_name
                and candidate.evidence_type is evidence_type
                and candidate.case_id_hint in {None, case_id}
                and (
                    candidate.value != expected_value
                    or candidate.applicant_hint not in {None, active_applicant}
                )
                for candidate in candidates
            ):
                return True
        return False

    @classmethod
    def _xw1_multisource_complete_review_recovery(
        cls,
        *,
        final_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_outcome: AdjudicationOutcome,
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence] = (),
        rapid_resolved: ResolvedCase | None = None,
    ) -> PredictionRow:
        """Apply the audited conservative XW-1 multisource approval rule.

        The rule is deliberately lower precedence than every denial or signed
        decision.  It accepts only one fully populated final review shape and
        requires three exact-case facts across two independent structured
        source types, each tied to the active applicant on the same page.
        """

        primary_candidates = tuple(
            candidate
            for candidate in primary_candidates
            if isinstance(candidate, CandidateEvidence)
        )
        rapid_candidates = tuple(
            candidate
            for candidate in rapid_candidates
            if isinstance(candidate, CandidateEvidence)
        )
        all_candidates = primary_candidates + rapid_candidates
        trace = primary_outcome.trace
        active_applicant = primary_resolved.active_applicant
        if (
            final_row.adjudication != "NEEDS_REVIEW"
            or primary_outcome.row.adjudication != "NEEDS_REVIEW"
            or trace.decision != "NEEDS_REVIEW"
            or primary_outcome.row.confidence > 0.25
            or final_row.case_id != primary_resolved.case_id
            or primary_outcome.row.case_id != primary_resolved.case_id
            or active_applicant is None
            or final_row.applicant_name != active_applicant
            or final_row.visa_class != "XW-1"
            or " ".join(final_row.risk_flags.strip().split()).casefold()
            != "none"
            or final_row.fee_status not in {"paid", "waived"}
            or not cls._complete_review_output(final_row)
            or trace.denial_reasons
            or cls._primary_authoritative_decision(primary_outcome)
            or primary_resolved.unresolved_linkage
            or primary_resolved.contested_fields
            or (
                rapid_resolved is not None
                and (
                    rapid_resolved.case_id != primary_resolved.case_id
                    or rapid_resolved.unresolved_linkage
                    or rapid_resolved.contested_fields
                    or cls._has_authoritative_rapid_decision(
                        case_id=primary_resolved.case_id,
                        primary_resolved=primary_resolved,
                        rapid_resolved=rapid_resolved,
                        rapid_candidates=rapid_candidates,
                    )
                )
            )
        ):
            return final_row

        facts = frozenset(trace.approval_facts)
        reasons = frozenset(trace.review_reasons)
        if not {
            "application_date_current_or_exempt",
            "sponsor_present_and_not_publicly_barred",
        }.issubset(facts):
            return final_row
        if final_row.fee_status == "paid":
            if (
                reasons
                != {
                    "required_output_unknown:risk_flags",
                    "risk_flags_unknown",
                }
                or "fee_paid" not in facts
            ):
                return final_row
        elif reasons != {"unsupported_fee_waiver"}:
            return final_row

        if any(
            RAPID_BAD_CUES.intersection(candidate.visual_cues)
            or candidate.evidence_type in AUTHORITATIVE_RAPID_TYPES
            for candidate in all_candidates
        ):
            return final_row
        if any(
            cls._clean_multisource_candidate(candidate)
            and candidate.field_name == RAPID_RISK_FIELD
            and " ".join(str(candidate.value).strip().split()).casefold()
            not in {"", "none", "unknown", "null"}
            for candidate in all_candidates
        ):
            return final_row

        expected = (
            (
                "visa_class",
                final_row.visa_class,
                EvidenceType.SPONSOR_ATTESTATION,
            ),
            (
                "home_world",
                final_row.home_world,
                EvidenceType.REGISTRY_EXTRACT,
            ),
            (
                "arrival_date",
                final_row.arrival_date,
                EvidenceType.REGISTRY_EXTRACT,
            ),
        )
        if cls._multisource_conflict(
            case_id=primary_resolved.case_id,
            active_applicant=active_applicant,
            expected=expected,
            candidates=all_candidates,
        ):
            return final_row
        if not all(
            cls._same_page_source_fact(
                case_id=primary_resolved.case_id,
                active_applicant=active_applicant,
                field_name=field_name,
                expected_value=expected_value,
                evidence_type=evidence_type,
                candidates=primary_candidates,
            )
            for field_name, expected_value, evidence_type in expected
        ):
            return final_row

        payload = final_row.to_dict()
        payload["adjudication"] = "APPROVED"
        payload["confidence"] = XW1_MULTISOURCE_REVIEW_APPROVAL_CONFIDENCE
        return PredictionRow.from_mapping(
            payload,
            fallback_case_id=final_row.case_id,
        )

    @classmethod
    def _apply_review_approval_heads(
        cls,
        *,
        final_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_outcome: AdjudicationOutcome,
        primary_resolved: ResolvedCase,
        rapid_candidates: Iterable[CandidateEvidence] = (),
        rapid_resolved: ResolvedCase | None = None,
    ) -> PredictionRow:
        """Run the conservative audited rule before the frozen broad head."""

        primary_candidates = tuple(primary_candidates)
        rapid_candidates = tuple(rapid_candidates)
        recovered = cls._xw1_multisource_complete_review_recovery(
            final_row=final_row,
            primary_candidates=primary_candidates,
            primary_outcome=primary_outcome,
            primary_resolved=primary_resolved,
            rapid_candidates=rapid_candidates,
            rapid_resolved=rapid_resolved,
        )
        return cls._review_approval_head(
            final_row=recovered,
            primary_candidates=primary_candidates,
            primary_outcome=primary_outcome,
            primary_resolved=primary_resolved,
            rapid_candidates=rapid_candidates,
            rapid_resolved=rapid_resolved,
        )

    @staticmethod
    def _repair_biometric_applicant(
        *,
        case_id: str,
        primary_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
    ) -> tuple[PredictionRow, bool]:
        """Prefer one stronger exact-case biometric name over one intake name.

        This is deliberately an output-only repair over evidence already read
        by the primary extractor.  It neither relinks the packet nor reruns
        resolution, policy, or calibration.  Any scope ambiguity abstains.
        """

        scoped = tuple(
            candidate
            for candidate in primary_candidates
            if isinstance(candidate, CandidateEvidence)
            and candidate.field_name == "applicant_name"
            and candidate.value is not None
            and candidate.evidence_type
            in {EvidenceType.BIOMETRIC_SLIP, EvidenceType.INTAKE_FORM}
            and candidate.source == "visible_ocr"
        )
        if any(
            candidate.case_id_hint not in {None, case_id}
            or RAPID_BAD_CUES.intersection(candidate.visual_cues)
            for candidate in scoped
        ):
            return primary_row, False

        relevant = tuple(
            candidate
            for candidate in scoped
            if candidate.legible and not candidate.superseded
        )

        exact_case = tuple(
            candidate
            for candidate in relevant
            if candidate.case_id_hint == case_id
        )
        biometrics = tuple(
            candidate
            for candidate in exact_case
            if candidate.evidence_type is EvidenceType.BIOMETRIC_SLIP
            and candidate.ocr_confidence
            >= BIOMETRIC_APPLICANT_MINIMUM_CONFIDENCE
        )
        intakes = tuple(
            candidate
            for candidate in exact_case
            if candidate.evidence_type is EvidenceType.INTAKE_FORM
        )
        biometric_values = {candidate.value for candidate in biometrics}
        intake_values = {candidate.value for candidate in intakes}
        if len(biometric_values) != 1 or len(intake_values) != 1:
            return primary_row, False

        biometric_value = next(iter(biometric_values))
        intake_value = next(iter(intake_values))
        if biometric_value == intake_value:
            return primary_row, False
        if max(candidate.ocr_confidence for candidate in biometrics) < max(
            candidate.ocr_confidence for candidate in intakes
        ):
            return primary_row, False

        payload = primary_row.to_dict()
        payload["applicant_name"] = biometric_value
        return (
            PredictionRow.from_mapping(payload, fallback_case_id=case_id),
            True,
        )

    @staticmethod
    def _safe_primary_candidate(candidate: object) -> bool:
        """Return whether one candidate is usable by output-only repairs."""

        return bool(
            isinstance(candidate, CandidateEvidence)
            and candidate.value is not None
            and candidate.legible
            and not candidate.superseded
            and candidate.source == "visible_ocr"
            and not RAPID_BAD_CUES.intersection(candidate.visual_cues)
        )

    @classmethod
    def _page_has_active_applicant(
        cls,
        *,
        case_id: str,
        page_index: int,
        active_applicant: str | None,
        candidates: Iterable[CandidateEvidence],
    ) -> bool:
        """Require one clean same-page applicant anchor when case ID is absent."""

        if active_applicant is None:
            return False
        return any(
            cls._safe_primary_candidate(candidate)
            and candidate.field_name == "applicant_name"
            and candidate.page_index == page_index
            and candidate.value == active_applicant
            and candidate.case_id_hint in {None, case_id}
            and candidate.applicant_hint in {None, active_applicant}
            for candidate in candidates
        )

    @classmethod
    def _repair_source_priority_fields(
        cls,
        *,
        case_id: str,
        primary_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
    ) -> tuple[PredictionRow, frozenset[str]]:
        """Repair three serialized fields without changing policy state.

        The frozen gates cover values redundantly visible on a sponsor or
        registry page when a noisier intake read won the binding precedence
        hierarchy.  This method changes only JSON output fields: resolution,
        adjudication, trace, and calibrated confidence stay untouched.
        """

        candidates = tuple(
            candidate
            for candidate in primary_candidates
            if isinstance(candidate, CandidateEvidence)
        )
        active_applicant = primary_resolved.active_applicant
        payload = primary_row.to_dict()
        repaired: set[str] = set()

        def intake_winner(field_name: str) -> CandidateEvidence | None:
            field = primary_resolved.fields.get(field_name)
            if (
                field is None
                or field.state is not FieldState.RESOLVED
                or field.value is None
                or payload.get(field_name) != field.value
                or not cls._safe_primary_candidate(field.winning_evidence)
                or field.winning_evidence.evidence_type
                is not EvidenceType.INTAKE_FORM
            ):
                return None
            return field.winning_evidence

        def unique_value(
            field_name: str,
            evidence_type: EvidenceType,
            allowed_cues: frozenset[str],
            scope: Callable[[CandidateEvidence], bool],
        ) -> tuple[str, tuple[CandidateEvidence, ...]] | None:
            eligible = tuple(
                candidate
                for candidate in candidates
                if cls._safe_primary_candidate(candidate)
                and candidate.field_name == field_name
                and candidate.evidence_type is evidence_type
                and candidate.ocr_confidence
                >= SOURCE_PRIORITY_MINIMUM_CONFIDENCE
                and set(candidate.visual_cues) <= allowed_cues
                and scope(candidate)
            )
            values = {candidate.value for candidate in eligible}
            if len(values) != 1:
                return None
            return next(iter(values)), eligible

        visa_winner = intake_winner("visa_class")
        if visa_winner is not None:
            visa = unique_value(
                "visa_class",
                EvidenceType.SPONSOR_ATTESTATION,
                frozenset({"structured_sponsor_narrative"}),
                lambda candidate: (
                    candidate.case_id_hint in {None, case_id}
                    and candidate.applicant_hint in {None, active_applicant}
                    and (
                        candidate.case_id_hint == case_id
                        or cls._page_has_active_applicant(
                            case_id=case_id,
                            page_index=candidate.page_index,
                            active_applicant=active_applicant,
                            candidates=candidates,
                        )
                    )
                ),
            )
            if visa is not None and visa[0] != payload["visa_class"]:
                payload["visa_class"] = visa[0]
                repaired.add("visa_class")

        sponsor_winner = intake_winner("sponsor_id")
        if sponsor_winner is not None and active_applicant is not None:
            sponsor = unique_value(
                "sponsor_id",
                EvidenceType.SPONSOR_ATTESTATION,
                frozenset({"structured_sponsor_narrative"}),
                lambda candidate: (
                    candidate.case_id_hint in {None, case_id}
                    and candidate.applicant_hint in {None, active_applicant}
                    and cls._page_has_active_applicant(
                        case_id=case_id,
                        page_index=candidate.page_index,
                        active_applicant=active_applicant,
                        candidates=candidates,
                    )
                ),
            )
            if (
                sponsor is not None
                and sponsor[0] != payload["sponsor_id"]
                and max(
                    candidate.ocr_confidence for candidate in sponsor[1]
                )
                > sponsor_winner.ocr_confidence
            ):
                payload["sponsor_id"] = sponsor[0]
                repaired.add("sponsor_id")

        arrival_winner = intake_winner("arrival_date")
        if arrival_winner is not None and active_applicant is not None:
            arrival = unique_value(
                "arrival_date",
                EvidenceType.REGISTRY_EXTRACT,
                frozenset(),
                lambda candidate: (
                    candidate.case_id_hint == case_id
                    and candidate.applicant_hint == active_applicant
                ),
            )
            if (
                arrival is not None
                and arrival[0] != payload["arrival_date"]
                and max(
                    candidate.ocr_confidence for candidate in arrival[1]
                )
                > arrival_winner.ocr_confidence
            ):
                payload["arrival_date"] = arrival[0]
                repaired.add("arrival_date")

        if not repaired:
            return primary_row, frozenset()
        return (
            PredictionRow.from_mapping(payload, fallback_case_id=case_id),
            frozenset(repaired),
        )

    def _recover(
        self,
        *,
        rendered: RenderedCase,
        primary_row: PredictionRow,
        primary_candidates: Iterable[CandidateEvidence],
        primary_resolved: ResolvedCase,
        primary_outcome: AdjudicationOutcome,
        unknown_fields: frozenset[str],
        recover_risk: bool,
    ) -> PredictionRow:
        rapid_candidates = tuple(self._rapid_extractor().extract(rendered))
        rapid_linked = self._linker.link(rendered.case_id, rapid_candidates)
        rapid_resolved = self._resolver.resolve(rapid_linked)
        payload = primary_row.to_dict()

        # Overlay only fields whose primary state is truly UNKNOWN.  Starting
        # from the primary row preserves its existing serialization priors
        # whenever Rapid is also unresolved.
        for field_name in unknown_fields:
            if field_name == "applicant_name":
                value = rapid_linked.active_applicant
            else:
                value = self._rapid_value(rapid_resolved, field_name)
            # ``unknown`` is a schema-valid fee literal, but it is not a
            # recovered fact.  Keep the primary serialization prior (``paid``)
            # when both OCR passes remain substantively unresolved.
            if value is not None and not (
                field_name == "fee_status" and value == "unknown"
            ):
                payload[field_name] = value

        if recover_risk:
            rapid_risk = self._rapid_value(rapid_resolved, RAPID_RISK_FIELD)
            if rapid_risk not in {None, "none"}:
                payload[RAPID_RISK_FIELD] = rapid_risk

        # Ordinary output recovery and the signed-decision override preserve
        # primary identity and calibration. The frozen semantic denial head
        # below is the only calibrated exception.
        payload["case_id"] = primary_row.case_id
        payload["confidence"] = primary_row.confidence

        decision = self._authoritative_rapid_decision(
            case_id=primary_resolved.case_id,
            primary_resolved=primary_resolved,
            primary_outcome=primary_outcome,
            rapid_candidates=rapid_candidates,
        )
        if decision is not None:
            payload["adjudication"] = decision
        elif self._semantic_denial_rules(
            case_id=primary_resolved.case_id,
            payload=payload,
            primary_candidates=primary_candidates,
            rapid_candidates=rapid_candidates,
            primary_resolved=primary_resolved,
            rapid_resolved=rapid_resolved,
            primary_outcome=primary_outcome,
            unknown_fields=unknown_fields,
            recover_risk=recover_risk,
        ):
            payload["adjudication"] = "DENIED"
            payload["confidence"] = SEMANTIC_DENIAL_CONFIDENCE
        final_row = PredictionRow.from_mapping(
            payload,
            fallback_case_id=primary_row.case_id,
        )
        return self._apply_review_approval_heads(
            final_row=final_row,
            primary_candidates=primary_candidates,
            primary_outcome=primary_outcome,
            primary_resolved=primary_resolved,
            rapid_candidates=rapid_candidates,
            rapid_resolved=rapid_resolved,
        )

    def process_case(self, pdf_path: Path) -> PredictionRow:
        rendered = self._renderer.render(pdf_path)
        primary_candidates = tuple(self._primary_extractor.extract(rendered))
        primary_linked = self._linker.link(rendered.case_id, primary_candidates)
        primary_resolved = self._resolver.resolve(primary_linked)
        primary_outcome = self._adjudicator.adjudicate_case(primary_resolved)

        primary_row, repaired_applicant = self._repair_biometric_applicant(
            case_id=rendered.case_id,
            primary_row=primary_outcome.row,
            primary_candidates=primary_candidates,
        )
        primary_row, source_repaired_fields = self._repair_source_priority_fields(
            case_id=rendered.case_id,
            primary_row=primary_row,
            primary_candidates=primary_candidates,
            primary_resolved=primary_resolved,
        )

        unknown_fields = self._unknown_output_fields(primary_resolved)
        if repaired_applicant:
            # The exact-case biometric fact is already frozen into the output;
            # an independent OCR pass must not replace it again.
            unknown_fields = unknown_fields - {"applicant_name"}
        unknown_fields = unknown_fields - source_repaired_fields
        recover_risk = self._recover_non_none_risk(
            primary_resolved,
            unknown_fields,
        )
        if not unknown_fields and not recover_risk:
            final_row = self._apply_review_approval_heads(
                final_row=primary_row,
                primary_candidates=primary_candidates,
                primary_outcome=primary_outcome,
                primary_resolved=primary_resolved,
            )
        else:
            try:
                final_row = self._recover(
                    rendered=rendered,
                    primary_row=primary_row,
                    primary_candidates=primary_candidates,
                    primary_resolved=primary_resolved,
                    primary_outcome=primary_outcome,
                    unknown_fields=unknown_fields,
                    recover_risk=recover_risk,
                )
            except Exception:
                # RapidOCR is optional recovery, never a reason to lose a primary
                # prediction or abort the batch.
                final_row = self._apply_review_approval_heads(
                    final_row=primary_row,
                    primary_candidates=primary_candidates,
                    primary_outcome=primary_outcome,
                    primary_resolved=primary_resolved,
                )
        # Identity-free layout-text field repairs (Amount/$809, registry name,
        # sponsor visa sentence). Never reads SYSTEM:/answer-key decoys.
        return apply_visible_field_repairs(final_row, pdf_path)
