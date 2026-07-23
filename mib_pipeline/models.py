"""Canonical prediction model and schema-safe value normalization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping


FIELD_NAMES = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
)

CASE_ID_PATTERN = re.compile(r"^MIB-[0-9]{6}$")
SPONSOR_ID_PATTERN = re.compile(r"^SPN-[0-9]{4}$")
FEE_VALUES = frozenset({"paid", "waived", "unpaid", "unknown"})
ADJUDICATION_VALUES = frozenset({"APPROVED", "DENIED", "NEEDS_REVIEW"})


class RowValidationError(ValueError):
    """Raised when no schema-valid prediction can be produced."""


def _safe_string(value: Any, fallback: str = "unknown") -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip()
    return normalized or fallback


def _safe_arrival_date(value: Any) -> str:
    if not isinstance(value, str):
        return "1900-01-01"
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError:
        return "1900-01-01"
    return normalized if parsed.isoformat() == normalized else "1900-01-01"


def _safe_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        return 0.0
    return normalized


@dataclass(frozen=True)
class PredictionRow:
    """Exactly the twelve fields accepted by submission.schema.json."""

    case_id: str
    applicant_name: str
    species_code: str
    home_world: str
    visa_class: str
    sponsor_id: str
    arrival_date: str
    declared_purpose: str
    risk_flags: str
    fee_status: str
    adjudication: str
    confidence: float

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        fallback_case_id: str | None = None,
    ) -> "PredictionRow":
        """Normalize computed values to conservative schema-valid values.

        A case ID cannot be fabricated safely. It must either be valid in the
        computed mapping or recoverable from the source PDF filename.
        """

        raw_case_id = value.get("case_id")
        case_id = raw_case_id.strip() if isinstance(raw_case_id, str) else ""
        if not CASE_ID_PATTERN.fullmatch(case_id):
            candidate = fallback_case_id.strip() if isinstance(fallback_case_id, str) else ""
            if not CASE_ID_PATTERN.fullmatch(candidate):
                raise RowValidationError("prediction has no recoverable case_id")
            case_id = candidate

        raw_sponsor_id = value.get("sponsor_id")
        sponsor_id = raw_sponsor_id.strip() if isinstance(raw_sponsor_id, str) else ""
        if not SPONSOR_ID_PATTERN.fullmatch(sponsor_id):
            sponsor_id = "SPN-0000"

        fee_status = _safe_string(value.get("fee_status"))
        if fee_status not in FEE_VALUES:
            fee_status = "unknown"

        adjudication = _safe_string(value.get("adjudication"), "NEEDS_REVIEW")
        if adjudication not in ADJUDICATION_VALUES:
            adjudication = "NEEDS_REVIEW"

        return cls(
            case_id=case_id,
            applicant_name=_safe_string(value.get("applicant_name")),
            species_code=_safe_string(value.get("species_code")),
            home_world=_safe_string(value.get("home_world")),
            visa_class=_safe_string(value.get("visa_class")),
            sponsor_id=sponsor_id,
            arrival_date=_safe_arrival_date(value.get("arrival_date")),
            declared_purpose=_safe_string(value.get("declared_purpose")),
            risk_flags=_safe_string(value.get("risk_flags"), "none"),
            fee_status=fee_status,
            adjudication=adjudication,
            confidence=_safe_confidence(value.get("confidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical field order with no additional keys."""

        return {field: getattr(self, field) for field in FIELD_NAMES}
