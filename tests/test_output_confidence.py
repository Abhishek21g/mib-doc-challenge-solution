import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from mib_pipeline import (
    FIELD_NAMES,
    OutputConfidenceArtifactError,
    OutputConfidenceRecalibrationProcessor,
    OutputConfidenceRecalibrator,
    PinnedOutputConfidenceMap,
    PredictionRow,
)
from mib_pipeline.output_confidence import (
    PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH,
    PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256,
)


def prediction(**overrides):
    value = {
        "case_id": "MIB-000001",
        "applicant_name": "Arix Vale",
        "species_code": "ARCTURIAN",
        "home_world": "Mars",
        "visa_class": "XW-1",
        "sponsor_id": "SPN-0001",
        "arrival_date": "2026-01-01",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.5,
    }
    value.update(overrides)
    return PredictionRow.from_mapping(value)


class FakeFinalProcessor:
    def __init__(self, row):
        self.row = row
        self.seen = []

    def process_case(self, pdf_path):
        self.seen.append(pdf_path)
        return self.row


class FrozenOutputConfidenceArtifactTests(unittest.TestCase):
    def test_pinned_artifact_is_exact_and_identity_free(self):
        payload = json.loads(
            PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH.read_text(encoding="utf-8")
        )
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        rendered = json.dumps(payload, sort_keys=True)

        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256,
        )
        self.assertEqual(payload["artifact_id"], "output-review-confidence-ridge-v1")
        self.assertEqual(len(payload["model"]["feature_order"]), 24)
        self.assertEqual(len(payload["model"]["coefficients"]), 24)
        self.assertEqual(
            payload["model"]["guard"],
            {
                "adjudication": "NEEDS_REVIEW",
                "maximum_exclusive_input_confidence": 0.9,
            },
        )
        self.assertIsNone(re.search(r"\bMIB-[0-9]{6}\b", rendered))
        self.assertIsNone(re.search(r"\bSPN-[0-9]{4}\b", rendered))
        self.assertIsNone(re.search(r"\.pdf\b", rendered, re.IGNORECASE))

    def test_pinned_loader_enforces_frozen_checksum(self):
        payload = json.loads(
            PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH.read_text(encoding="utf-8")
        )
        payload["model"]["coefficients"][0] += 0.001
        with tempfile.TemporaryDirectory() as temp_dir:
            changed_path = Path(temp_dir) / "changed.json"
            changed_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(OutputConfidenceArtifactError):
                PinnedOutputConfidenceMap.from_path(
                    changed_path,
                    expected_sha256=PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256,
                )

    def test_loader_rejects_identity_bearing_metadata(self):
        payload = json.loads(
            PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH.read_text(encoding="utf-8")
        )
        payload["identity_policy"] += " MIB-000001"

        with self.assertRaises(OutputConfidenceArtifactError):
            PinnedOutputConfidenceMap.from_mapping(payload)


class OutputConfidenceRecalibratorTests(unittest.TestCase):
    def setUp(self):
        self.recalibrator = OutputConfidenceRecalibrator.from_pinned_artifact()

    def test_exact_frozen_prediction_and_probability_bounds(self):
        recalibrated = self.recalibrator.recalibrate(prediction(confidence=0.5))
        lower_clipped = self.recalibrator.recalibrate(prediction(confidence=0.0))

        self.assertAlmostEqual(recalibrated.confidence, 0.5587794014582416)
        self.assertEqual(lower_clipped.confidence, 0.02)
        self.assertTrue(0.02 <= recalibrated.confidence <= 0.98)

    def test_guard_is_review_only_and_threshold_is_exclusive(self):
        for adjudication in ("APPROVED", "DENIED"):
            with self.subTest(adjudication=adjudication):
                row = prediction(adjudication=adjudication, confidence=0.1)
                self.assertIs(self.recalibrator.recalibrate(row), row)

        at_threshold = prediction(confidence=0.9)
        above_threshold = prediction(confidence=0.98)
        self.assertIs(self.recalibrator.recalibrate(at_threshold), at_threshold)
        self.assertIs(self.recalibrator.recalibrate(above_threshold), above_threshold)
        self.assertNotEqual(
            self.recalibrator.recalibrate(prediction(confidence=0.899)).confidence,
            0.899,
        )

    def test_only_confidence_changes_byte_equivalent_fields(self):
        row = prediction(
            applicant_name="Quorin Ax",
            species_code="CENTAURI_SYNTH",
            home_world="Europa Station",
            visa_class="MED-3",
            sponsor_id="SPN-4312",
            arrival_date="2026-04-09",
            declared_purpose="medical consult",
            risk_flags="memory_tampering",
            fee_status="waived",
            confidence=0.4,
        )
        recalibrated = self.recalibrator.recalibrate(row)
        before = row.to_dict()
        after = recalibrated.to_dict()

        self.assertNotEqual(after["confidence"], before["confidence"])
        self.assertEqual(
            {field: before[field] for field in FIELD_NAMES if field != "confidence"},
            {field: after[field] for field in FIELD_NAMES if field != "confidence"},
        )

    def test_identity_values_do_not_affect_recalibration(self):
        first = prediction(
            case_id="MIB-000001",
            applicant_name="Arix Vale",
            home_world="Mars",
            sponsor_id="SPN-0001",
            arrival_date="2026-01-01",
        )
        second = prediction(
            case_id="MIB-999999",
            applicant_name="Different Person",
            species_code="DIFFERENT_SPECIES",
            home_world="A Different World",
            visa_class="DIFFERENT-VISA",
            sponsor_id="SPN-9999",
            arrival_date="2026-12-31",
            declared_purpose="different purpose",
        )

        self.assertEqual(
            self.recalibrator.recalibrate(first).confidence,
            self.recalibrator.recalibrate(second).confidence,
        )

    def test_processor_runs_after_final_decision_and_preserves_non_reviews(self):
        final_approval = prediction(adjudication="APPROVED", confidence=0.2)
        approval_inner = FakeFinalProcessor(final_approval)
        processor = OutputConfidenceRecalibrationProcessor(
            processor=approval_inner,
            recalibrator=self.recalibrator,
        )
        source = Path("MIB-000001.pdf")

        self.assertIs(processor.process_case(source), final_approval)
        self.assertEqual(approval_inner.seen, [source])

        final_review = prediction(confidence=0.2)
        review_processor = OutputConfidenceRecalibrationProcessor(
            processor=FakeFinalProcessor(final_review),
            recalibrator=self.recalibrator,
        )
        self.assertNotEqual(
            review_processor.process_case(source).confidence,
            final_review.confidence,
        )


if __name__ == "__main__":
    unittest.main()
