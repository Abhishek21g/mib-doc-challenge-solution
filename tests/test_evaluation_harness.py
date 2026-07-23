import csv
import json
import tempfile
import unittest
from pathlib import Path

from devtools.evaluation import (
    EvaluationConfigurationError,
    EvaluationHarness,
    HoldoutSplitManager,
    IsotonicCalibrationFitter,
    PolicyExceptionValidator,
    ReleaseGate,
    RunMetrics,
    RuntimeArtifactLeakageScanner,
    SplitEvidence,
)
from mib_pipeline import (
    GeneralizablePolicyExceptionStore,
    PinnedIsotonicMap,
    PolicyArtifactError,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
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
)


def truth_row(case_id, adjudication):
    return {
        "case_id": case_id,
        "applicant_name": "Zed Zarnax",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1042",
        "arrival_date": "2026-04-17",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": adjudication,
    }


def prediction(row, adjudication=None, confidence=0.8):
    return {
        **row,
        "adjudication": adjudication or row["adjudication"],
        "confidence": confidence,
    }


class HoldoutSplitTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_complete_and_disjoint(self):
        rows = [
            truth_row(f"MIB-{index:06d}", decisions[index % 3])
            for index in range(1, 31)
            for decisions in [("APPROVED", "DENIED", "NEEDS_REVIEW")]
        ]
        first = HoldoutSplitManager(seed="repro-v1").split_rows(rows)
        second = HoldoutSplitManager(seed="repro-v1").split_rows(reversed(rows))

        self.assertEqual(first, second)
        self.assertEqual(sum(len(values) for values in first.values()), 30)
        self.assertFalse(set(first["tuning"]) & set(first["calibration"]))
        self.assertFalse(set(first["tuning"]) & set(first["release"]))
        self.assertTrue(all(first[role] for role in first))

    def test_overlap_and_self_tuned_measurement_are_rejected(self):
        with self.assertRaises(EvaluationConfigurationError):
            HoldoutSplitManager.assert_no_overlap(
                ["MIB-000001"], ["MIB-000001"]
            )
        with self.assertRaises(EvaluationConfigurationError):
            SplitEvidence(
                name="release-v1",
                role="release",
                tuned_on_splits=("release-v1",),
            )

    def test_previously_inspected_cases_are_forced_into_tuning(self):
        rows = [
            truth_row(f"MIB-{index:06d}", ("APPROVED", "DENIED")[index % 2])
            for index in range(1, 41)
        ]
        forced = {"MIB-000001", "MIB-000002", "MIB-000003"}

        splits = HoldoutSplitManager(seed="repro-v2").split_rows(
            rows,
            forced_tuning_case_ids=forced,
        )

        self.assertTrue(forced <= set(splits["tuning"]))
        self.assertFalse(forced & set(splits["calibration"]))
        self.assertFalse(forced & set(splits["release"]))

    def test_unknown_forced_tuning_case_is_rejected(self):
        with self.assertRaises(EvaluationConfigurationError):
            HoldoutSplitManager(seed="repro-v2").split_rows(
                [truth_row("MIB-000001", "APPROVED")],
                forced_tuning_case_ids=("MIB-999999",),
            )


class EvaluationHarnessTests(unittest.TestCase):
    def test_wraps_official_evaluator_with_required_breakdowns_and_metrics(self):
        truths = [
            truth_row("MIB-000001", "DENIED"),
            truth_row("MIB-000002", "APPROVED"),
            truth_row("MIB-000003", "NEEDS_REVIEW"),
        ]
        predictions = [
            prediction(truths[0], "APPROVED", 0.9),
            prediction(truths[1], confidence=0.8),
            prediction(truths[2], confidence=0.7),
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            truth_path = directory_path / "truth.csv"
            submission_path = directory_path / "submission.jsonl"
            with truth_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(truths)
            submission_path.write_text(
                "".join(json.dumps(row) + "\n" for row in predictions),
                encoding="utf-8",
            )

            report = EvaluationHarness(REPO_ROOT).evaluate(
                truth_path=truth_path,
                submission_path=submission_path,
                split=SplitEvidence(
                    "release-v1", "release", tuned_on_splits=("tuning-v1",)
                ),
                run_metrics=RunMetrics(12.5, 256.0, "offline_runtime_measurement"),
                golden_case_ids=("MIB-000002",),
                adversarial_case_ids=("MIB-000001",),
            )

        self.assertEqual(
            report["benchmark_context"],
            "local_engineering_benchmark_not_official_leaderboard",
        )
        self.assertEqual(report["summary"]["catastrophic_false_approvals"], 1)
        self.assertEqual(report["summary"]["missing_rows"], 0)
        self.assertEqual(report["summary"]["invalid_rows"], 0)
        self.assertEqual(report["measurements"]["runtime_seconds"], 12.5)
        self.assertEqual(
            set(report["per_field_match_rates"]),
            {
                "applicant_name",
                "species_code",
                "home_world",
                "visa_class",
                "sponsor_id",
                "arrival_date",
                "declared_purpose",
                "risk_flags",
                "fee_status",
            },
        )
        self.assertEqual(
            report["per_field_match_rates"]["species_code"]["match_rate"], 1.0
        )
        self.assertTrue(report["case_outcomes"]["MIB-000002"]["golden"])
        self.assertTrue(report["case_outcomes"]["MIB-000001"]["adversarial"])
        for score in (
            "total_score",
            "extraction_score",
            "classification_score",
            "calibration_score",
        ):
            self.assertIn(score, report["summary"])


def release_report(*, false_approvals=0, honest=True, split_name="release-v1"):
    return {
        "split": {
            "name": split_name,
            "role": "release" if honest else "tuning",
            "is_honest_holdout": honest,
        },
        "summary": {
            "total_score": 100.0,
            "extraction_score": 40.0,
            "classification_score": 50.0,
            "calibration_score": 10.0,
            "catastrophic_false_approvals": false_approvals,
            "missing_rows": 0,
            "invalid_rows": 0,
        },
        "measurements": {
            "runtime_seconds": 100.0,
            "peak_memory_mib": 500.0,
            "source": "offline_runtime_measurement",
        },
        "case_outcomes": {
            "MIB-000001": {
                "correct": True,
                "golden": True,
                "adversarial": False,
            },
            "MIB-000002": {
                "correct": True,
                "golden": False,
                "adversarial": True,
            },
        },
        "leakage_findings": [],
    }


class ReleaseGateTests(unittest.TestCase):
    def test_blocks_false_approval_increase(self):
        decision = ReleaseGate().decide(
            baseline=release_report(false_approvals=0),
            candidate=release_report(false_approvals=1),
        )

        self.assertEqual(decision["decision"], "BLOCKED")
        self.assertFalse(decision["adopt"])
        self.assertIn(
            "catastrophic false approvals increased", decision["blocking_reasons"]
        )

    def test_blocks_tuning_data_mismatched_split_leakage_and_missing_metrics(self):
        candidate = release_report(honest=False, split_name="tuning-v1")
        candidate["leakage_findings"] = ["case ID lookup"]
        candidate.pop("measurements")
        decision = ReleaseGate().decide(
            baseline=release_report(), candidate=candidate
        )

        self.assertEqual(decision["decision"], "BLOCKED")
        self.assertGreaterEqual(len(decision["blocking_reasons"]), 4)

    def test_flags_new_golden_and_adversarial_regressions(self):
        candidate = release_report()
        candidate["case_outcomes"]["MIB-000001"]["correct"] = False
        candidate["case_outcomes"]["MIB-000002"]["correct"] = False
        decision = ReleaseGate().decide(
            baseline=release_report(), candidate=candidate
        )

        self.assertEqual(decision["decision"], "PASS_WITH_WARNINGS")
        self.assertTrue(decision["adopt"])
        self.assertEqual(decision["golden_regressions"], ["MIB-000001"])
        self.assertEqual(decision["adversarial_regressions"], ["MIB-000002"])


class ArtifactHygieneTests(unittest.TestCase):
    def test_leakage_scanner_rejects_identity_keys_and_values(self):
        for value in (
            {"case_id": "anything"},
            {"feature": "MIB-000001"},
            {"feature": "packet.pdf"},
            {"feature": "a" * 64},
        ):
            with self.subTest(value=value), self.assertRaises(
                EvaluationConfigurationError
            ):
                RuntimeArtifactLeakageScanner.require_clean(value)

    def test_isotonic_fitter_uses_honest_split_and_pools_violations(self):
        split = SplitEvidence(
            "calibration-v1", "calibration", tuned_on_splits=("tuning-v1",)
        )
        samples = [
            {
                "case_id": "MIB-000001",
                "split_name": "calibration-v1",
                "raw_signal": 0.2,
                "correct": True,
            },
            {
                "case_id": "MIB-000002",
                "split_name": "calibration-v1",
                "raw_signal": 0.4,
                "correct": False,
            },
            {
                "case_id": "MIB-000003",
                "split_name": "calibration-v1",
                "raw_signal": 0.8,
                "correct": True,
            },
        ]
        artifact = IsotonicCalibrationFitter.fit(
            samples,
            artifact_id="honest-isotonic-v1",
            split=split,
            calibration_case_ids={
                "MIB-000001",
                "MIB-000002",
                "MIB-000003",
            },
            tuning_case_ids={"MIB-000999"},
        )
        mapping = PinnedIsotonicMap.from_mapping(artifact)

        self.assertEqual(mapping.probabilities, tuple(sorted(mapping.probabilities)))
        self.assertEqual(artifact["fit_metadata"]["training_case_count"], 3)
        self.assertFalse(RuntimeArtifactLeakageScanner.findings(artifact))

    def test_isotonic_fitter_rejects_tuning_and_mixed_samples(self):
        sample = {
            "case_id": "MIB-000001",
            "split_name": "calibration-v1",
            "raw_signal": 0.5,
            "correct": True,
        }
        with self.assertRaises(EvaluationConfigurationError):
            IsotonicCalibrationFitter.fit(
                [sample],
                artifact_id="bad",
                split=SplitEvidence("tuning-v1", "tuning"),
                calibration_case_ids={"MIB-000001"},
                tuning_case_ids={"MIB-000999"},
            )
        with self.assertRaises(EvaluationConfigurationError):
            IsotonicCalibrationFitter.fit(
                [sample],
                artifact_id="bad",
                split=SplitEvidence("other-calibration", "calibration"),
                calibration_case_ids={"MIB-000001"},
                tuning_case_ids={"MIB-000999"},
            )

    def test_policy_exception_validator_accepts_only_heldout_safe_rules(self):
        candidates = [
            {
                "rule_id": "other-barred-sponsor",
                "conditions": {"sponsor_id": "SPN-9999"},
                "decision": "DENIED",
                "rationale": "visible sponsor revocation generalizes",
            },
            {
                "rule_id": "identity-leak",
                "conditions": {"case_id": "MIB-000001"},
                "decision": "DENIED",
                "rationale": "memorized",
            },
        ]
        evidence = {
            "other-barred-sponsor": {
                "visible_support_case_ids": [
                    "MIB-000001",
                    "MIB-000002",
                    "MIB-000003",
                ],
                "heldout_support_case_ids": ["MIB-000002", "MIB-000003"],
                "tuning_case_ids": ["MIB-000001"],
                "tuning_split": "tuning-v1",
                "measured_split": "release-v1",
                "baseline_false_approvals": 0,
                "candidate_false_approvals": 0,
            },
            "identity-leak": {
                "visible_support_case_ids": [
                    "MIB-000004",
                    "MIB-000005",
                    "MIB-000006",
                ],
                "heldout_support_case_ids": ["MIB-000005", "MIB-000006"],
                "tuning_case_ids": ["MIB-000004"],
                "tuning_split": "tuning-v1",
                "measured_split": "release-v1",
                "baseline_false_approvals": 0,
                "candidate_false_approvals": 0,
            },
        }
        artifact, rejected = PolicyExceptionValidator().validate(
            candidates, evidence, artifact_id="validated-exceptions-v2"
        )

        self.assertEqual(
            [rule["rule_id"] for rule in artifact["exceptions"]],
            ["other-barred-sponsor"],
        )
        self.assertIn("identity-leak", rejected)
        self.assertFalse(RuntimeArtifactLeakageScanner.findings(artifact))

    def test_runtime_loads_only_the_pinned_validated_exception_artifact(self):
        store = GeneralizablePolicyExceptionStore.from_pinned_artifact()

        self.assertEqual(store.artifact_id, "validated-policy-exceptions-v1")

    def test_runtime_rejects_unsafe_policy_exception_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_exceptions.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_id": "unsafe",
                        "exceptions": [
                            {
                                "rule_id": "leak",
                                "conditions": {"case_id": "MIB-000001"},
                                "decision": "DENIED",
                                "rationale": "memorized",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(PolicyArtifactError):
                GeneralizablePolicyExceptionStore.from_pinned_artifact(path)


class BoundaryTests(unittest.TestCase):
    def test_submission_dockerfile_does_not_copy_development_harness(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertNotIn("devtools", dockerfile)
        self.assertNotIn("evaluation_harness.py", dockerfile)
        self.assertNotIn("data/", dockerfile)


if __name__ == "__main__":
    unittest.main()
