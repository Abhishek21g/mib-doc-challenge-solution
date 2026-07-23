"""Frozen confidence-only recalibration over the final serialized output.

The postprocessor is intentionally downstream of OCR recovery and all final
decision overrides.  Its guard admits only low-confidence ``NEEDS_REVIEW``
rows, and its typed mutation uses :func:`dataclasses.replace` to make every
non-confidence field immutable at the model boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from .confidence import CalibrationArtifactError
from .models import PredictionRow


PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "output_confidence_recalibration.json"
)
PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256 = (
    "51d00d8d6dcf309fe6a33543d6ffbc7858914b7549d853b71b4489594046dd1c"
)

_OUTPUT_FIELDS = (
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
_FROZEN_FEATURE_ORDER = (
    "bias",
    "p",
    "p2",
    "output=APPROVED",
    "output=DENIED",
    "output=NEEDS_REVIEW",
    "p_output=APPROVED",
    "p_output=DENIED",
    "p_output=NEEDS_REVIEW",
    "unknown=applicant_name",
    "unknown=species_code",
    "unknown=home_world",
    "unknown=visa_class",
    "unknown=sponsor_id",
    "unknown=arrival_date",
    "unknown=declared_purpose",
    "unknown=risk_flags",
    "unknown=fee_status",
    "unknown_count",
    "risk_non_none",
    "fee_paid",
    "fee_waived",
    "fee_unpaid",
    "fee_unknown",
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bMIB-[0-9]{6}\b|\bSPN-[0-9]{4}\b|\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b|\.pdf\b)",
    re.IGNORECASE,
)


class OutputConfidenceArtifactError(CalibrationArtifactError):
    """The final-output confidence artifact is malformed or not frozen."""


class FinalPredictionProcessor(Protocol):
    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        """Return the final typed row after all output and decision recovery."""


def _contains_sensitive_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_sensitive_value(key) or _contains_sensitive_value(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_value(child) for child in value)
    return isinstance(value, str) and _SENSITIVE_VALUE.search(value) is not None


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _is_unknown(field_name: str, value: Any) -> bool:
    rendered = str(value or "").strip().casefold()
    if rendered in {"", "unknown", "null"}:
        return True
    if field_name == "sponsor_id" and rendered == "spn-0000":
        return True
    return field_name == "arrival_date" and rendered == "1900-01-01"


@dataclass(frozen=True)
class PinnedOutputConfidenceMap:
    """Validated frozen ridge map over identity-free output semantics."""

    artifact_id: str
    coefficients: tuple[float, ...]
    feature_order: tuple[str, ...]
    blend: float
    probability_clip: tuple[float, float]
    guarded_adjudication: str
    maximum_exclusive_input_confidence: float
    fit_metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PinnedOutputConfidenceMap":
        if set(value) != {
            "artifact_id",
            "schema_version",
            "fit_metadata",
            "model",
            "identity_policy",
        } or value.get("schema_version") != 1:
            raise OutputConfidenceArtifactError(
                "unsupported output-confidence artifact schema"
            )
        artifact_id = value.get("artifact_id")
        metadata = value.get("fit_metadata")
        identity_policy = value.get("identity_policy")
        model = value.get("model")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise OutputConfidenceArtifactError("artifact_id must be non-empty")
        if not isinstance(metadata, dict) or not isinstance(identity_policy, str):
            raise OutputConfidenceArtifactError("artifact metadata is malformed")
        if _contains_sensitive_value(value):
            raise OutputConfidenceArtifactError(
                "output-confidence artifact contains identity-bearing values"
            )
        if not isinstance(model, dict) or set(model) != {
            "ridge_strength",
            "blend",
            "probability_clip",
            "guard",
            "feature_order",
            "coefficients",
        }:
            raise OutputConfidenceArtifactError(
                "output-confidence model is malformed"
            )
        ridge_strength = model["ridge_strength"]
        blend = model["blend"]
        if not _is_number(ridge_strength) or float(ridge_strength) <= 0:
            raise OutputConfidenceArtifactError("ridge_strength must be positive")
        if not _is_number(blend) or not 0.0 <= float(blend) <= 1.0:
            raise OutputConfidenceArtifactError("blend must be within [0,1]")

        raw_clip = model["probability_clip"]
        if (
            not isinstance(raw_clip, list)
            or len(raw_clip) != 2
            or any(not _is_number(item) for item in raw_clip)
        ):
            raise OutputConfidenceArtifactError("probability_clip is malformed")
        lower, upper = (float(item) for item in raw_clip)
        if not 0.0 <= lower < upper <= 1.0:
            raise OutputConfidenceArtifactError("probability_clip is not ordered")

        guard = model["guard"]
        if not isinstance(guard, dict) or set(guard) != {
            "adjudication",
            "maximum_exclusive_input_confidence",
        }:
            raise OutputConfidenceArtifactError("confidence guard is malformed")
        if guard["adjudication"] != "NEEDS_REVIEW" or not _is_number(
            guard["maximum_exclusive_input_confidence"]
        ):
            raise OutputConfidenceArtifactError("confidence guard is unsupported")
        maximum_confidence = float(guard["maximum_exclusive_input_confidence"])
        if not 0.0 <= maximum_confidence <= 1.0:
            raise OutputConfidenceArtifactError("confidence guard is out of bounds")

        raw_features = model["feature_order"]
        if not isinstance(raw_features, list) or tuple(raw_features) != _FROZEN_FEATURE_ORDER:
            raise OutputConfidenceArtifactError(
                "feature order is not the frozen identity-free feature set"
            )
        raw_coefficients = model["coefficients"]
        if (
            not isinstance(raw_coefficients, list)
            or len(raw_coefficients) != len(_FROZEN_FEATURE_ORDER)
            or any(not _is_number(item) for item in raw_coefficients)
        ):
            raise OutputConfidenceArtifactError("coefficients are malformed")

        return cls(
            artifact_id=artifact_id,
            coefficients=tuple(float(item) for item in raw_coefficients),
            feature_order=_FROZEN_FEATURE_ORDER,
            blend=float(blend),
            probability_clip=(lower, upper),
            guarded_adjudication="NEEDS_REVIEW",
            maximum_exclusive_input_confidence=maximum_confidence,
            fit_metadata=dict(metadata),
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> "PinnedOutputConfidenceMap":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OutputConfidenceArtifactError(
                f"cannot load output-confidence artifact: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise OutputConfidenceArtifactError(
                "output-confidence artifact must be an object"
            )
        if expected_sha256 is not None and _canonical_sha256(value) != expected_sha256:
            raise OutputConfidenceArtifactError(
                "output-confidence artifact does not match the frozen checksum"
            )
        return cls.from_mapping(value)

    @staticmethod
    def _feature_values(row: PredictionRow) -> dict[str, float]:
        probability = float(row.confidence)
        unknown = {
            field_name: _is_unknown(field_name, getattr(row, field_name))
            for field_name in _OUTPUT_FIELDS
        }
        values = {
            "bias": 1.0,
            "p": probability,
            "p2": probability * probability,
            f"output={row.adjudication}": 1.0,
            f"p_output={row.adjudication}": probability,
            "unknown_count": sum(unknown.values()) / len(_OUTPUT_FIELDS),
            "risk_non_none": float(
                row.risk_flags.strip().casefold()
                not in {"", "none", "unknown", "null"}
            ),
            f"fee_{row.fee_status.strip().casefold()}": 1.0,
        }
        values.update(
            {
                f"unknown={field_name}": float(is_unknown)
                for field_name, is_unknown in unknown.items()
            }
        )
        return values

    def predict(self, row: PredictionRow) -> float:
        """Return the original confidence unless the frozen guard admits it."""

        if (
            row.adjudication != self.guarded_adjudication
            or row.confidence >= self.maximum_exclusive_input_confidence
        ):
            return row.confidence
        values = self._feature_values(row)
        raw_probability = sum(
            values.get(feature, 0.0) * coefficient
            for feature, coefficient in zip(
                self.feature_order,
                self.coefficients,
                strict=True,
            )
        )
        probability = row.confidence + self.blend * (
            raw_probability - row.confidence
        )
        lower, upper = self.probability_clip
        return max(lower, min(upper, probability))


class OutputConfidenceRecalibrator:
    """Apply the frozen map while mutating only ``PredictionRow.confidence``."""

    def __init__(self, calibration_map: PinnedOutputConfidenceMap) -> None:
        self._map = calibration_map

    @classmethod
    def from_pinned_artifact(
        cls,
        path: Path = PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH,
    ) -> "OutputConfidenceRecalibrator":
        expected_sha256 = (
            PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256
            if path == PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH
            else None
        )
        return cls(
            PinnedOutputConfidenceMap.from_path(
                path,
                expected_sha256=expected_sha256,
            )
        )

    @property
    def artifact_id(self) -> str:
        return self._map.artifact_id

    def recalibrate(self, row: PredictionRow) -> PredictionRow:
        confidence = self._map.predict(row)
        if confidence == row.confidence:
            return row
        return replace(row, confidence=confidence)


@dataclass
class OutputConfidenceRecalibrationProcessor:
    """Outermost production stage over the fully recovered prediction row."""

    processor: FinalPredictionProcessor
    recalibrator: OutputConfidenceRecalibrator

    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        row = self.processor.process_case(pdf_path)
        if row is None:
            return None
        if not isinstance(row, PredictionRow):
            raise TypeError("output-confidence postprocessor requires PredictionRow")
        return self.recalibrator.recalibrate(row)
