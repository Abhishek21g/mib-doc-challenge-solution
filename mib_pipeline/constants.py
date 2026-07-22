from __future__ import annotations

from datetime import date

FIELDNAMES = [
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
]

VISA_CLASSES = {"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"}
FEE_STATUSES = {"paid", "waived", "unpaid", "unknown"}
ADJUDICATIONS = {"APPROVED", "DENIED", "NEEDS_REVIEW"}

# Closed vocabularies from the public field manual / train distribution.
KNOWN_PURPOSES = {
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
}
KNOWN_SPECIES = {
    "TRIANGULAN",
    "JOVIAN_GASFORM",
    "CENTAURI_SYNTH",
    "LUNA_SECURID",
    "KAIJU_MICRO",
    "ORION_GRAYS",
    "ALPHA_DRACONIAN",
    "SIRIUS_AVIAN",
    "VENUSIAN_MYCELIAL",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "ANDROMEDAN",
}
KNOWN_WORLDS = {
    "Luyten-b",
    "Europa Station",
    "Titan Freeport",
    "Barnard-c",
    "Gliese-581g",
    "Mars Dome-7",
    "Kepler-186f",
    "Sirius Outpost",
    "Wolf-1061c",
    "Proxima-b",
    "Zeta Reticuli",
    "TRAPPIST-1e",
    "Eris Relay",
}

DISQUALIFYING_FLAGS = {
    "memory_tampering",
    "planetary_embargo",
    "active_warrant",
    "biohazard_red",
}
REVIEW_FLAGS = {
    "identity_conflict",
    "sponsor_mismatch",
    "illegible_biometrics",
    "rescinded_denial",
}
ALL_FLAGS = DISQUALIFYING_FLAGS | REVIEW_FLAGS

REVOKED_SPONSORS = {
    "SPN-0007",
    "SPN-0139",
    "SPN-4040",
    "SPN-9090",
    "SPN-7331",
    "SPN-2718",
}

ALWAYS_EMBARGO_WORLDS = {"TRAPPIST-1e", "Eris Relay"}
NONDIP_EMBARGO_WORLDS = {"Wolf-1061c"}

# Challenge epoch used when packet receipt date is not printed.
DEFAULT_RECEIPT_DATE = date(2026, 7, 1)

# Span trust filters (reject white / decoy prompt-injection text).
MAX_TRUSTED_COLOR = 16_700_000
MIN_TRUSTED_SIZE = 5.5

DOC_MARKERS = {
    "intake": "FORM I-8090",
    "fee_receipt": "MIB Fee Receipt",
    "biometric": "FORM B-13",
    "sponsor_letter": "Sponsor Attestation",
    "registry": "Planetary Registry",
    "adjudicator_note": "Manual Adjudicator",
}

LABEL_TO_FIELD = {
    "applicant": "applicant_name",
    "registry name": "applicant_name",
    "species code": "species_code",
    "species match": "species_code",
    "home world": "home_world",
    "visa class": "visa_class",
    "sponsor id": "sponsor_id",
    "arrival date": "arrival_date",
    "declared purpose": "declared_purpose",
    "fee status": "fee_status",
    "observed flags": "observed_flags",
    "registry status": "registry_status",
    "waiver code": "waiver_code",
    "case id": "case_id_field",
}

# Source precedence for identity / policy fields (lower = higher trust).
SOURCE_RANK = {
    "adjudicator_note": 0,
    "intake": 1,
    "biometric": 2,
    "sponsor_letter": 3,
    "registry": 4,
    "fee_receipt": 5,
    "unknown": 6,
    "ocr": 7,
    # OCR of a known form type — better than bare ocr, worse than native text.
    "ocr_intake": 8,
    "ocr_biometric": 8,
    "ocr_sponsor_letter": 8,
    "ocr_registry": 8,
    "ocr_fee_receipt": 8,
    "ocr_adjudicator_note": 8,
}
