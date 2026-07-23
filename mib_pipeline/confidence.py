"""Decision-correctness signals and pinned confidence calibration.

The runtime artifact supports the original one-dimensional isotonic map and a
semantic hierarchical model.  The latter is deliberately keyed only by
generic policy-trace features.  Values that can identify an applicant or a
case (for example a sponsor ID or a home-world value) are redacted before a
lookup key is formed.
"""

from __future__ import annotations

import json
import math
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adjudication import DecisionTrace


PINNED_ARTIFACT_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "confidence_calibration.json"
)


class CalibrationArtifactError(ValueError):
    """The pinned calibration artifact is absent, malformed, or unsafe."""


_DECISIONS = frozenset({"APPROVED", "DENIED", "NEEDS_REVIEW"})
_SAFE_FEATURE_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")
_SENSITIVE_ARTIFACT_VALUE = re.compile(
    r"(?:\bMIB-[0-9]{6}\b|\bSPN-[0-9]{4}\b|\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b|\.pdf\b)",
    re.IGNORECASE,
)

# A suffix is retained only when it describes a fixed policy category or a
# schema field.  Suffixes on all other reasons are replaced by ``*``.  This is
# what prevents values such as ``SPN-0007`` or ``TRAPPIST-1e`` from becoming
# artifact keys while preserving useful distinctions such as the risk class.
_SAFE_SUFFIX_PREFIXES = frozenset(
    {
        "contested_field",
        "disqualifying_flag",
        "required_output_not_visible",
        "required_output_unknown",
        "review_flag",
    }
)

_KNOWN_BARE_FEATURES = frozenset(
    {
        "application_date_current_or_exempt",
        "arrival_date_not_visible",
        "arrival_date_unknown",
        "authoritative_visible_decision",
        "biohazard_red",
        "clean_biohazard_check",
        "clean_biohazard_check_missing",
        "conflicting_generalizable_exceptions",
        "diplomatic_sponsor_exemption",
        "fee_paid",
        "fee_status_unknown",
        "no_visible_biohazard_risk",
        "packet_receipt_date_unknown",
        "required_sponsor_absent",
        "required_sponsor_not_visible",
        "required_sponsor_unknown",
        "risk_flags_not_visible",
        "risk_flags_unknown",
        "sponsor_present_and_not_publicly_barred",
        "stale_application",
        "stale_diplomatic_note_exemption",
        "stale_diplomatic_note_missing",
        "stay_duration_unknown",
        "stay_within_visa_limit",
        "strict_approval_bar_cleared",
        "transit_work_authorization",
        "unpaid_without_valid_waiver",
        "unsupported_fee_waiver",
        "valid_fee_waiver",
        "validated_generalizable_exception",
        "visa_class_not_visible",
        "visa_class_unknown",
    }
)


def _canonical_feature(value: str) -> str:
    """Return an auditable, identity-free policy feature token."""

    rendered = str(value).strip().casefold()
    if ":" not in rendered:
        return rendered if rendered in _KNOWN_BARE_FEATURES else "other"
    prefix, suffix = rendered.split(":", 1)
    if not _SAFE_FEATURE_TOKEN.fullmatch(prefix):
        return "other"
    if (
        prefix in _SAFE_SUFFIX_PREFIXES
        and _SAFE_FEATURE_TOKEN.fullmatch(suffix) is not None
    ):
        return f"{prefix}:{suffix}"
    return f"{prefix}:*"


def _semantic_features(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({_canonical_feature(value) for value in values}))


def _primary_bucket(trace: DecisionTrace) -> str:
    """Select one compact semantic parent bucket for a decision trace."""

    decision = trace.decision
    if decision not in _DECISIONS:
        raise ValueError(f"unsupported adjudication in decision trace: {decision}")
    if trace.authoritative_source:
        return f"{decision}|authoritative"

    if decision == "DENIED":
        ordered_categories = (
            "disqualifying_flag:",
            "barred_sponsor:",
            "embargoed_home_world:",
            "transit_work_authorization",
            "biohazard_red",
            "unpaid_without_valid_waiver",
            "stale_application",
            "stay_limit_exceeded:",
            "required_sponsor_absent",
            "validated_generalizable_exception",
        )
        for category in ordered_categories:
            if any(reason.startswith(category) for reason in trace.denial_reasons):
                return f"{decision}|reason={category.rstrip(':')}"
        return f"{decision}|reason=other"

    if decision == "APPROVED":
        support_count = sum(
            fact != "strict_approval_bar_cleared" for fact in trace.approval_facts
        )
        return f"{decision}|support={min(6, support_count)}"

    review_reasons = trace.review_reasons
    if any(reason.startswith("review_flag:") for reason in review_reasons):
        review_class = "review_flag"
    elif any(
        reason.startswith(
            (
                "contested_field:",
                "unresolved_linkage:",
                "conflicting_generalizable_exceptions",
            )
        )
        for reason in review_reasons
    ):
        review_class = "conflict"
    elif any(
        reason.startswith(
            (
                "required_output_",
                "visa_class_",
                "risk_flags_",
                "fee_status_",
                "arrival_date_",
                "required_sponsor_",
            )
        )
        or "not_visible" in reason
        for reason in review_reasons
    ):
        review_class = "core_gap"
    else:
        review_class = "policy_only"
    return f"{decision}|class={review_class}"


def _trace_signature(trace: DecisionTrace) -> str:
    """Build an exact signature after removing all case-specific values."""

    source = "authoritative" if trace.authoritative_source else "rules"
    denial = ",".join(_semantic_features(trace.denial_reasons))
    review = ",".join(_semantic_features(trace.review_reasons))
    approval = ",".join(_semantic_features(trace.approval_facts))
    return (
        f"{trace.decision}|source={source}|denial={denial}|"
        f"review={review}|approval={approval}"
    )


def _is_primary_bucket_key(value: str) -> bool:
    decision, separator, feature = value.partition("|")
    if not separator or decision not in _DECISIONS:
        return False
    if feature == "authoritative":
        return True
    if decision == "DENIED":
        return feature.startswith("reason=") and bool(
            _SAFE_FEATURE_TOKEN.fullmatch(feature[len("reason=") :])
        )
    if decision == "APPROVED":
        return feature in {f"support={index}" for index in range(7)}
    return feature in {
        "class=review_flag",
        "class=conflict",
        "class=core_gap",
        "class=policy_only",
    }


def _is_trace_signature_key(value: str) -> bool:
    sections = value.split("|")
    if len(sections) != 5 or sections[0] not in _DECISIONS:
        return False
    if sections[1] not in {"source=authoritative", "source=rules"}:
        return False
    for section, prefix in zip(
        sections[2:], ("denial=", "review=", "approval="), strict=True
    ):
        if not section.startswith(prefix):
            return False
        rendered_features = section[len(prefix) :]
        if not rendered_features:
            continue
        features = rendered_features.split(",")
        if features != sorted(set(features)):
            return False
        if any(_canonical_feature(feature) != feature for feature in features):
            return False
    return True


def _contains_sensitive_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_sensitive_value(key) or _contains_sensitive_value(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_value(child) for child in value)
    return isinstance(value, str) and _SENSITIVE_ARTIFACT_VALUE.search(value) is not None


@dataclass(frozen=True)
class _BinomialStatistics:
    correct: int
    total: int

    @classmethod
    def from_value(cls, value: Any, *, label: str) -> "_BinomialStatistics":
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            raise CalibrationArtifactError(f"{label} must be [correct, total] integers")
        correct, total = value
        if total <= 0 or correct < 0 or correct > total:
            raise CalibrationArtifactError(f"{label} contains invalid binomial counts")
        return cls(correct=correct, total=total)


@dataclass(frozen=True)
class PinnedSemanticMap:
    """Hierarchical empirical-Bayes map over identity-free trace semantics."""

    artifact_id: str
    global_statistics: _BinomialStatistics
    decision_statistics: Mapping[str, _BinomialStatistics]
    bucket_statistics: Mapping[str, _BinomialStatistics]
    signature_statistics: Mapping[str, _BinomialStatistics]
    decision_strength: float
    bucket_strength: float
    signature_strength: float
    probability_clip: tuple[float, float]
    fit_metadata: Mapping[str, Any]

    @staticmethod
    def _statistics_table(value: Any, *, label: str) -> dict[str, _BinomialStatistics]:
        if not isinstance(value, dict) or not value:
            raise CalibrationArtifactError(f"{label} must be a non-empty object")
        result: dict[str, _BinomialStatistics] = {}
        for raw_key, raw_statistics in value.items():
            if not isinstance(raw_key, str) or not raw_key or len(raw_key) > 4096:
                raise CalibrationArtifactError(f"{label} contains an invalid key")
            if _SENSITIVE_ARTIFACT_VALUE.search(raw_key):
                raise CalibrationArtifactError(f"{label} contains identity-bearing values")
            result[raw_key] = _BinomialStatistics.from_value(
                raw_statistics, label=f"{label}.{raw_key}"
            )
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PinnedSemanticMap":
        allowed_keys = {"schema_version", "artifact_id", "model", "fit_metadata"}
        if set(value) != allowed_keys or value.get("schema_version") != 2:
            raise CalibrationArtifactError("unsupported calibration artifact schema")
        artifact_id = value.get("artifact_id")
        model = value.get("model")
        metadata = value.get("fit_metadata")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise CalibrationArtifactError("artifact_id must be a non-empty string")
        if not isinstance(model, dict) or set(model) != {
            "global_statistics",
            "decision_statistics",
            "primary_bucket_statistics",
            "trace_signature_statistics",
            "smoothing_strengths",
            "probability_clip",
        }:
            raise CalibrationArtifactError("semantic calibration model is malformed")
        if not isinstance(metadata, dict):
            raise CalibrationArtifactError("fit_metadata must be an object")
        if _contains_sensitive_value(artifact_id) or _contains_sensitive_value(metadata):
            raise CalibrationArtifactError(
                "calibration artifact contains identity-bearing values"
            )

        global_statistics = _BinomialStatistics.from_value(
            model["global_statistics"], label="global_statistics"
        )
        decisions = cls._statistics_table(
            model["decision_statistics"], label="decision_statistics"
        )
        buckets = cls._statistics_table(
            model["primary_bucket_statistics"], label="primary_bucket_statistics"
        )
        signatures = cls._statistics_table(
            model["trace_signature_statistics"], label="trace_signature_statistics"
        )
        if set(decisions) != _DECISIONS:
            raise CalibrationArtifactError("decision statistics must cover all decisions")
        if any(not _is_primary_bucket_key(key) for key in buckets):
            raise CalibrationArtifactError(
                "primary bucket key is not identity-free semantics"
            )
        if any(not _is_trace_signature_key(key) for key in signatures):
            raise CalibrationArtifactError(
                "trace signature key is not identity-free semantics"
            )

        smoothing = model["smoothing_strengths"]
        if not isinstance(smoothing, dict) or set(smoothing) != {
            "decision_to_global",
            "bucket_to_decision",
            "signature_to_bucket",
        }:
            raise CalibrationArtifactError("smoothing strengths are malformed")
        strengths: list[float] = []
        for key in (
            "decision_to_global",
            "bucket_to_decision",
            "signature_to_bucket",
        ):
            raw_strength = smoothing[key]
            if isinstance(raw_strength, bool) or not isinstance(raw_strength, (int, float)):
                raise CalibrationArtifactError("smoothing strengths must be numeric")
            strength = float(raw_strength)
            if not math.isfinite(strength) or strength <= 0:
                raise CalibrationArtifactError("smoothing strengths must be positive")
            strengths.append(strength)

        raw_clip = model["probability_clip"]
        if not isinstance(raw_clip, list) or len(raw_clip) != 2:
            raise CalibrationArtifactError("probability_clip must contain two values")
        try:
            lower, upper = (float(item) for item in raw_clip)
        except (TypeError, ValueError) as exc:
            raise CalibrationArtifactError("probability_clip must be numeric") from exc
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 <= lower < upper <= 1):
            raise CalibrationArtifactError("probability_clip must be ordered within [0,1]")

        expected = (global_statistics.correct, global_statistics.total)
        for label, table in (("decision", decisions),):
            observed = (
                sum(statistics.correct for statistics in table.values()),
                sum(statistics.total for statistics in table.values()),
            )
            if observed != expected:
                raise CalibrationArtifactError(
                    f"{label} statistics do not reconcile with global statistics"
                )
        for label, table in (
            ("primary bucket", buckets),
            ("trace signature", signatures),
        ):
            for decision, parent in decisions.items():
                children = [
                    statistics
                    for key, statistics in table.items()
                    if key.startswith(f"{decision}|")
                ]
                observed = (
                    sum(statistics.correct for statistics in children),
                    sum(statistics.total for statistics in children),
                )
                if observed != (parent.correct, parent.total):
                    raise CalibrationArtifactError(
                        f"{label} statistics do not reconcile with {decision}"
                    )

        return cls(
            artifact_id=artifact_id,
            global_statistics=global_statistics,
            decision_statistics=decisions,
            bucket_statistics=buckets,
            signature_statistics=signatures,
            decision_strength=strengths[0],
            bucket_strength=strengths[1],
            signature_strength=strengths[2],
            probability_clip=(lower, upper),
            fit_metadata=dict(metadata),
        )

    def _smoothed_probability(
        self,
        statistics: _BinomialStatistics,
        *,
        parent_probability: float,
        strength: float,
    ) -> float:
        return (
            statistics.correct + strength * parent_probability
        ) / (statistics.total + strength)

    def predict(self, trace: DecisionTrace) -> float:
        if trace.decision not in _DECISIONS:
            raise ValueError(
                f"unsupported adjudication in decision trace: {trace.decision}"
            )
        global_probability = (
            self.global_statistics.correct / self.global_statistics.total
        )
        decision_probability = self._smoothed_probability(
            self.decision_statistics[trace.decision],
            parent_probability=global_probability,
            strength=self.decision_strength,
        )
        bucket_probability = decision_probability
        bucket = self.bucket_statistics.get(_primary_bucket(trace))
        if bucket is not None:
            bucket_probability = self._smoothed_probability(
                bucket,
                parent_probability=decision_probability,
                strength=self.bucket_strength,
            )
        probability = bucket_probability
        signature = self.signature_statistics.get(_trace_signature(trace))
        if signature is not None:
            probability = self._smoothed_probability(
                signature,
                parent_probability=bucket_probability,
                strength=self.signature_strength,
            )
        lower, upper = self.probability_clip
        return max(lower, min(upper, probability))


@dataclass(frozen=True)
class PinnedIsotonicMap:
    """A validated monotonic step map fitted outside the submitted runtime."""

    artifact_id: str
    breakpoints: tuple[float, ...]
    probabilities: tuple[float, ...]
    fit_metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PinnedIsotonicMap":
        allowed_keys = {
            "schema_version",
            "artifact_id",
            "breakpoints",
            "probabilities",
            "fit_metadata",
        }
        if set(value) != allowed_keys or value.get("schema_version") != 1:
            raise CalibrationArtifactError("unsupported calibration artifact schema")
        artifact_id = value.get("artifact_id")
        breakpoints = value.get("breakpoints")
        probabilities = value.get("probabilities")
        metadata = value.get("fit_metadata")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise CalibrationArtifactError("artifact_id must be a non-empty string")
        if not isinstance(breakpoints, list) or not isinstance(probabilities, list):
            raise CalibrationArtifactError("isotonic arrays must be JSON lists")
        if len(breakpoints) != len(probabilities) or len(breakpoints) < 2:
            raise CalibrationArtifactError("isotonic arrays must have equal useful length")
        if not isinstance(metadata, dict):
            raise CalibrationArtifactError("fit_metadata must be an object")
        try:
            xs = tuple(float(item) for item in breakpoints)
            ys = tuple(float(item) for item in probabilities)
        except (TypeError, ValueError) as exc:
            raise CalibrationArtifactError("isotonic values must be numeric") from exc
        if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in xs + ys):
            raise CalibrationArtifactError("isotonic values must be finite in [0,1]")
        if any(right <= left for left, right in zip(xs, xs[1:])):
            raise CalibrationArtifactError("isotonic breakpoints must strictly increase")
        if any(right < left for left, right in zip(ys, ys[1:])):
            raise CalibrationArtifactError("isotonic probabilities must not decrease")
        if xs[0] != 0.0 or xs[-1] != 1.0:
            raise CalibrationArtifactError("isotonic breakpoints must span [0,1]")
        return cls(
            artifact_id=artifact_id,
            breakpoints=xs,
            probabilities=ys,
            fit_metadata=dict(metadata),
        )

    @classmethod
    def from_path(cls, path: Path) -> "PinnedIsotonicMap":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationArtifactError(
                f"cannot load calibration artifact: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise CalibrationArtifactError("calibration artifact must be an object")
        return cls.from_mapping(value)

    def predict(self, raw_signal: float) -> float:
        if not math.isfinite(raw_signal):
            raise ValueError("raw decision signal must be finite")
        bounded = max(0.0, min(1.0, float(raw_signal)))
        index = max(0, bisect_right(self.breakpoints, bounded) - 1)
        return self.probabilities[index]


def _load_pinned_map(path: Path) -> PinnedIsotonicMap | PinnedSemanticMap:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationArtifactError(f"cannot load calibration artifact: {path}") from exc
    if not isinstance(value, dict):
        raise CalibrationArtifactError("calibration artifact must be an object")
    if value.get("schema_version") == 1:
        return PinnedIsotonicMap.from_mapping(value)
    if value.get("schema_version") == 2:
        return PinnedSemanticMap.from_mapping(value)
    raise CalibrationArtifactError("unsupported calibration artifact schema")


class DecisionSignalModel:
    """Estimate decision strength from policy semantics, never OCR confidence."""

    _HARD_DENIAL_PREFIXES = (
        "barred_sponsor:",
        "disqualifying_flag:",
        "embargoed_home_world:",
        "stay_limit_exceeded:",
    )
    _HARD_DENIAL_REASONS = frozenset(
        {
            "biohazard_red",
            "required_sponsor_absent",
            "stale_application",
            "transit_work_authorization",
            "unpaid_without_valid_waiver",
        }
    )

    @staticmethod
    def _has_prefix(reasons: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
        return any(reason.startswith(prefixes) for reason in reasons)

    def raw_signal(self, trace: DecisionTrace) -> float:
        """Return a deterministic pre-calibration P(correct-decision) signal."""

        if trace.authoritative_source:
            return 0.98

        if trace.decision == "DENIED":
            hard_reason = bool(set(trace.denial_reasons) & self._HARD_DENIAL_REASONS)
            hard_reason = hard_reason or self._has_prefix(
                trace.denial_reasons, self._HARD_DENIAL_PREFIXES
            )
            base = 0.93 if hard_reason else 0.84
            corroboration = min(0.05, max(0, len(trace.denial_reasons) - 1) * 0.015)
            return min(0.99, base + corroboration)

        if trace.decision == "NEEDS_REVIEW":
            reasons = trace.review_reasons
            if self._has_prefix(reasons, ("contested_field:",)):
                base = 0.94
            elif self._has_prefix(reasons, ("unresolved_linkage:",)):
                base = 0.92
            elif self._has_prefix(reasons, ("review_flag:",)):
                base = 0.88
            elif any("not_visible" in reason for reason in reasons):
                base = 0.83
            else:
                base = 0.74
            corroboration = min(0.06, max(0, len(reasons) - 1) * 0.01)
            return min(0.98, base + corroboration)

        if trace.decision == "APPROVED":
            supporting_facts = len(
                [
                    fact
                    for fact in trace.approval_facts
                    if fact != "strict_approval_bar_cleared"
                ]
            )
            return min(0.93, 0.78 + min(6, supporting_facts) * 0.025)

        raise ValueError(f"unsupported adjudication in decision trace: {trace.decision}")


class ConfidenceCalibrator:
    """Map a policy decision trace to P(the emitted adjudication is correct)."""

    def __init__(
        self,
        calibration_map: PinnedIsotonicMap | PinnedSemanticMap,
        *,
        signal_model: DecisionSignalModel | None = None,
    ) -> None:
        self._map = calibration_map
        self._signal_model = signal_model or DecisionSignalModel()

    @classmethod
    def from_pinned_artifact(
        cls,
        path: Path = PINNED_ARTIFACT_PATH,
    ) -> "ConfidenceCalibrator":
        return cls(_load_pinned_map(path))

    @property
    def artifact_id(self) -> str:
        return self._map.artifact_id

    def raw_signal(self, trace: DecisionTrace) -> float:
        return self._signal_model.raw_signal(trace)

    def calibrate(self, trace: DecisionTrace) -> float:
        """Return bounded calibrated confidence for the chosen decision."""

        if isinstance(self._map, PinnedSemanticMap):
            return self._map.predict(trace)
        return self._map.predict(self.raw_signal(trace))
