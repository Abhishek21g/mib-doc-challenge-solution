import math
import json
import re
import unittest
from pathlib import Path

from mib_pipeline import (
    CalibrationArtifactError,
    ConfidenceCalibrator,
    DecisionSignalModel,
    DecisionTrace,
    PinnedIsotonicMap,
)
from mib_pipeline.confidence import (
    PinnedSemanticMap,
    _primary_bucket,
    _trace_signature,
)


def trace(
    decision,
    *,
    authoritative=False,
    denial=(),
    review=(),
    approval=(),
):
    return DecisionTrace(
        decision=decision,
        authoritative_source=authoritative,
        denial_reasons=tuple(denial),
        review_reasons=tuple(review),
        approval_facts=tuple(approval),
        exception_ids=(),
    )


def artifact(**overrides):
    value = {
        "schema_version": 1,
        "artifact_id": "test-map-v1",
        "breakpoints": [0.0, 0.5, 0.75, 1.0],
        "probabilities": [0.2, 0.55, 0.8, 0.95],
        "fit_metadata": {
            "method": "isotonic_regression",
            "target": "emitted_adjudication_is_correct",
        },
    }
    value.update(overrides)
    return value


def semantic_artifact(**overrides):
    samples = (
        (
            trace(
                "APPROVED",
                approval=("fee_paid", "strict_approval_bar_cleared"),
            ),
            True,
        ),
        (
            trace("DENIED", denial=("barred_sponsor:SPN-0007",)),
            True,
        ),
        (
            trace(
                "NEEDS_REVIEW",
                review=("required_output_unknown:risk_flags", "risk_flags_unknown"),
            ),
            False,
        ),
    )

    def statistics(key):
        grouped = {}
        for sample_trace, correct in samples:
            rendered = key(sample_trace)
            counts = grouped.setdefault(rendered, [0, 0])
            counts[0] += int(correct)
            counts[1] += 1
        return grouped

    value = {
        "schema_version": 2,
        "artifact_id": "test-semantic-map-v2",
        "model": {
            "global_statistics": [2, 3],
            "decision_statistics": statistics(lambda item: item.decision),
            "primary_bucket_statistics": statistics(_primary_bucket),
            "trace_signature_statistics": statistics(_trace_signature),
            "smoothing_strengths": {
                "decision_to_global": 8,
                "bucket_to_decision": 2,
                "signature_to_bucket": 2,
            },
            "probability_clip": [0.02, 0.98],
        },
        "fit_metadata": {
            "method": "hierarchical_empirical_bayes_semantic_trace",
            "target": "emitted_adjudication_is_correct",
        },
    }
    value.update(overrides)
    return value


class PinnedIsotonicMapTests(unittest.TestCase):
    def test_pinned_artifact_loads_and_spans_probability_range(self):
        calibrator = ConfidenceCalibrator.from_pinned_artifact()

        self.assertTrue(calibrator.artifact_id)
        values = [
            calibrator.calibrate(
                trace("DENIED", denial=("disqualifying_flag:active_warrant",))
            ),
            calibrator.calibrate(
                trace("NEEDS_REVIEW", review=("fee_status_unknown",))
            ),
            calibrator.calibrate(
                trace(
                    "APPROVED",
                    approval=(
                        "fee_paid",
                        "sponsor_present_and_not_publicly_barred",
                        "strict_approval_bar_cleared",
                    ),
                )
            ),
        ]

        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertGreater(len(set(values)), 1)

    def test_calibrated_review_confidence_is_not_a_fixed_default(self):
        calibrator = ConfidenceCalibrator.from_pinned_artifact()
        missing = calibrator.calibrate(
            trace("NEEDS_REVIEW", review=("fee_status_unknown",))
        )
        contested = calibrator.calibrate(
            trace(
                "NEEDS_REVIEW",
                review=("contested_field:visa_class", "fee_status_unknown"),
            )
        )

        self.assertNotEqual(missing, contested)

    def test_isotonic_prediction_is_monotonic_and_bounded(self):
        mapping = PinnedIsotonicMap.from_mapping(artifact())
        predictions = [mapping.predict(value / 20) for value in range(-5, 26)]

        self.assertEqual(predictions, sorted(predictions))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in predictions))
        self.assertEqual(mapping.predict(-100), 0.2)
        self.assertEqual(mapping.predict(100), 0.95)

    def test_rejects_malformed_or_non_monotonic_artifacts(self):
        malformed = (
            artifact(breakpoints=[0.0, 0.8, 0.7, 1.0]),
            artifact(probabilities=[0.2, 0.8, 0.7, 0.95]),
            artifact(breakpoints=[0.1, 0.5, 0.75, 1.0]),
            artifact(probabilities=[0.2, math.nan, 0.8, 0.95]),
            {**artifact(), "unexpected": True},
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(CalibrationArtifactError):
                PinnedIsotonicMap.from_mapping(value)


class PinnedSemanticMapTests(unittest.TestCase):
    def test_semantic_hierarchy_uses_signature_then_bucket_then_decision(self):
        mapping = PinnedSemanticMap.from_mapping(semantic_artifact())
        fitted_signature = trace(
            "NEEDS_REVIEW",
            review=("required_output_unknown:risk_flags", "risk_flags_unknown"),
        )
        unseen_signature_in_fitted_bucket = trace(
            "NEEDS_REVIEW",
            review=("required_output_unknown:fee_status", "fee_status_unknown"),
        )

        fitted = mapping.predict(fitted_signature)
        bucket_fallback = mapping.predict(unseen_signature_in_fitted_bucket)

        self.assertLess(fitted, bucket_fallback)
        self.assertTrue(0.02 <= fitted <= 0.98)
        self.assertTrue(0.02 <= bucket_fallback <= 0.98)

    def test_case_specific_reason_values_collapse_to_one_signature(self):
        first = trace("DENIED", denial=("barred_sponsor:SPN-0007",))
        second = trace("DENIED", denial=("barred_sponsor:SPN-9090",))
        first_world = trace("DENIED", denial=("embargoed_home_world:Eris Relay",))
        second_world = trace(
            "DENIED", denial=("embargoed_home_world:TRAPPIST-1e",)
        )

        self.assertEqual(_trace_signature(first), _trace_signature(second))
        self.assertEqual(_primary_bucket(first), _primary_bucket(second))
        self.assertEqual(_trace_signature(first_world), _trace_signature(second_world))

    def test_rejects_inconsistent_or_identity_bearing_semantic_artifacts(self):
        inconsistent = semantic_artifact()
        inconsistent["model"]["global_statistics"] = [1, 3]
        identity_key = semantic_artifact()
        signatures = identity_key["model"]["trace_signature_statistics"]
        old_key = next(iter(signatures))
        signatures[f"{old_key}|case=MIB-000001"] = signatures.pop(old_key)
        invalid_strength = semantic_artifact()
        invalid_strength["model"]["smoothing_strengths"][
            "signature_to_bucket"
        ] = math.nan

        for value in (inconsistent, identity_key, invalid_strength):
            with self.subTest(value=value), self.assertRaises(CalibrationArtifactError):
                PinnedSemanticMap.from_mapping(value)

    def test_pinned_runtime_artifact_contains_no_identity_values(self):
        artifact_path = (
            Path(__file__).resolve().parents[1]
            / "mib_pipeline"
            / "artifacts"
            / "confidence_calibration.json"
        )
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        rendered = json.dumps(payload, sort_keys=True)

        self.assertEqual(payload["schema_version"], 2)
        self.assertIsNone(re.search(r"\bMIB-[0-9]{6}\b", rendered, re.IGNORECASE))
        self.assertIsNone(re.search(r"\bSPN-[0-9]{4}\b", rendered, re.IGNORECASE))
        self.assertIsNone(re.search(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b", rendered))


class DecisionSignalTests(unittest.TestCase):
    def setUp(self):
        self.model = DecisionSignalModel()

    def test_authoritative_and_hard_denial_signals_are_strong(self):
        authoritative = self.model.raw_signal(
            trace("APPROVED", authoritative=True, approval=("source",))
        )
        hard_denial = self.model.raw_signal(
            trace("DENIED", denial=("disqualifying_flag:active_warrant",))
        )

        self.assertGreaterEqual(authoritative, 0.95)
        self.assertGreaterEqual(hard_denial, 0.9)

    def test_review_confidence_reflects_why_review_is_correct(self):
        missing = self.model.raw_signal(
            trace("NEEDS_REVIEW", review=("fee_status_unknown",))
        )
        untrusted = self.model.raw_signal(
            trace(
                "NEEDS_REVIEW",
                review=("fee_status_not_visible", "risk_flags_not_visible"),
            )
        )
        contested = self.model.raw_signal(
            trace(
                "NEEDS_REVIEW",
                review=("contested_field:visa_class", "fee_status_unknown"),
            )
        )

        self.assertNotEqual(missing, untrusted)
        self.assertGreater(contested, untrusted)

    def test_approval_signal_depends_on_policy_support_not_ocr_score(self):
        lightly_supported = self.model.raw_signal(
            trace(
                "APPROVED",
                approval=("strict_approval_bar_cleared", "fee_paid"),
            )
        )
        strongly_supported = self.model.raw_signal(
            trace(
                "APPROVED",
                approval=(
                    "strict_approval_bar_cleared",
                    "fee_paid",
                    "sponsor_present_and_not_publicly_barred",
                    "application_date_current_or_exempt",
                    "stay_within_visa_limit",
                ),
            )
        )

        self.assertGreater(strongly_supported, lightly_supported)

    def test_unknown_decision_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.raw_signal(trace("MAYBE"))


if __name__ == "__main__":
    unittest.main()
