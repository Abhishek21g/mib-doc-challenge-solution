"""Honest evaluation, calibration fitting, and safety release gating."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FIELD_NAMES = (
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
NON_TUNING_ROLES = frozenset({"calibration", "release", "golden", "adversarial"})


class EvaluationConfigurationError(ValueError):
    """Evaluation inputs violate holdout, artifact, or release-gate contracts."""


@dataclass(frozen=True)
class RunMetrics:
    runtime_seconds: float
    peak_memory_mib: float
    source: str

    def __post_init__(self) -> None:
        values = (self.runtime_seconds, self.peak_memory_mib)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise EvaluationConfigurationError("runtime metrics must be finite and non-negative")
        if not self.source:
            raise EvaluationConfigurationError("runtime metric source is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_mib": self.peak_memory_mib,
            "source": self.source,
        }


@dataclass(frozen=True)
class SplitEvidence:
    name: str
    role: str
    tuned_on_splits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.role:
            raise EvaluationConfigurationError("split name and role are required")
        if self.name in self.tuned_on_splits:
            raise EvaluationConfigurationError(
                "a measured split cannot also be declared as a tuning split"
            )

    @property
    def is_honest_holdout(self) -> bool:
        return self.role in NON_TUNING_ROLES and self.name not in self.tuned_on_splits

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "tuned_on_splits": list(self.tuned_on_splits),
            "is_honest_holdout": self.is_honest_holdout,
        }


class HoldoutSplitManager:
    """Create deterministic stratified splits and prove their separation."""

    ROLES = ("tuning", "calibration", "release")

    def __init__(
        self,
        *,
        seed: str,
        tuning_fraction: float = 0.7,
        calibration_fraction: float = 0.15,
    ) -> None:
        if not seed:
            raise EvaluationConfigurationError("a non-empty split seed is required")
        if not 0 < tuning_fraction < 1:
            raise EvaluationConfigurationError("tuning_fraction must be in (0,1)")
        if not 0 < calibration_fraction < 1 - tuning_fraction:
            raise EvaluationConfigurationError(
                "calibration_fraction must leave cases for release"
            )
        self._seed = seed
        self._tuning_fraction = tuning_fraction
        self._calibration_fraction = calibration_fraction

    def _key(self, case_id: str) -> str:
        payload = f"{self._seed}\0{case_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def split_rows(
        self,
        rows: Iterable[Mapping[str, str]],
        *,
        forced_tuning_case_ids: Iterable[str] = (),
    ) -> dict[str, tuple[str, ...]]:
        forced_tuning = {
            str(case_id).strip()
            for case_id in forced_tuning_case_ids
            if str(case_id).strip()
        }
        grouped: dict[str, list[str]] = defaultdict(list)
        seen: set[str] = set()
        for row in rows:
            case_id = str(row.get("case_id", "")).strip()
            adjudication = str(row.get("adjudication", "UNKNOWN")).strip().upper()
            if not case_id or case_id in seen:
                raise EvaluationConfigurationError("split rows need unique non-empty case IDs")
            seen.add(case_id)
            grouped[adjudication].append(case_id)

        unknown_forced = forced_tuning - seen
        if unknown_forced:
            raise EvaluationConfigurationError(
                "forced tuning cases are absent from truth: "
                + ", ".join(sorted(unknown_forced)[:5])
            )

        splits: dict[str, list[str]] = {role: [] for role in self.ROLES}
        for case_ids in grouped.values():
            forced_in_group = sorted(set(case_ids) & forced_tuning)
            eligible = sorted(set(case_ids) - forced_tuning, key=self._key)
            tuning_target = round(len(case_ids) * self._tuning_fraction)
            tuning_count = max(0, tuning_target - len(forced_in_group))
            calibration_count = round(
                len(case_ids) * self._calibration_fraction
            )
            tuning_end = min(tuning_count, len(eligible))
            calibration_end = min(tuning_end + calibration_count, len(eligible))
            splits["tuning"].extend(forced_in_group)
            splits["tuning"].extend(eligible[:tuning_end])
            splits["calibration"].extend(eligible[tuning_end:calibration_end])
            splits["release"].extend(eligible[calibration_end:])

        result = {role: tuple(sorted(ids)) for role, ids in splits.items()}
        self.assert_disjoint(result)
        return result

    @staticmethod
    def assert_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
        owners: dict[str, str] = {}
        for split_name, case_ids in splits.items():
            for case_id in case_ids:
                previous = owners.setdefault(case_id, split_name)
                if previous != split_name:
                    raise EvaluationConfigurationError(
                        f"case appears in multiple splits: {case_id}"
                    )

    @staticmethod
    def assert_no_overlap(
        measured_case_ids: Iterable[str],
        tuning_case_ids: Iterable[str],
    ) -> None:
        overlap = set(measured_case_ids) & set(tuning_case_ids)
        if overlap:
            raise EvaluationConfigurationError(
                f"honest holdout overlaps tuning data by {len(overlap)} cases"
            )


class RuntimeArtifactLeakageScanner:
    """Reject case identity, filenames, and hashes from runtime artifacts."""

    _FORBIDDEN_KEYS = frozenset(
        {
            "case_id",
            "case_ids",
            "filename",
            "filenames",
            "pdf_hash",
            "pdf_hashes",
            "file_hash",
            "sha256",
        }
    )

    @classmethod
    def findings(cls, value: Any, path: str = "$") -> tuple[str, ...]:
        findings: list[str] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized_key = str(key).strip().casefold()
                child_path = f"{path}.{key}"
                if normalized_key in cls._FORBIDDEN_KEYS:
                    findings.append(f"{child_path}: forbidden identity key")
                findings.extend(cls.findings(child, child_path))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                findings.extend(cls.findings(child, f"{path}[{index}]"))
        elif isinstance(value, str):
            rendered = value.strip()
            if rendered.lower().endswith(".pdf"):
                findings.append(f"{path}: filename value")
            if rendered.upper().startswith("MIB-") and rendered[4:].isdigit():
                findings.append(f"{path}: case-ID value")
            if 32 <= len(rendered) <= 64 and all(
                character in "0123456789abcdefABCDEF" for character in rendered
            ):
                findings.append(f"{path}: hash-like value")
        return tuple(findings)

    @classmethod
    def require_clean(cls, value: Any) -> None:
        findings = cls.findings(value)
        if findings:
            raise EvaluationConfigurationError(
                "runtime artifact contains case-identity leakage: " + "; ".join(findings)
            )


def _peak_memory_mib(raw_value: float) -> float:
    # macOS reports bytes; Linux reports KiB.
    return raw_value / (1024 * 1024) if sys.platform == "darwin" else raw_value / 1024


class EvaluationHarness:
    """Wrap the official evaluator and add engineering/safety breakdowns."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()
        self._evaluator = self._repo_root / "scripts" / "evaluate.py"
        if not self._evaluator.is_file():
            raise EvaluationConfigurationError("official evaluator script is missing")

    @staticmethod
    def _field_rates(case_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rates: dict[str, Any] = {}
        for field_name in FIELD_NAMES:
            matched = 0
            scorable = 0
            for case in case_scores:
                result = case["field_results"][field_name]
                if result["status"] == "not_scorable_unrecoverable":
                    continue
                scorable += 1
                matched += int(result["status"] == "matched")
            rates[field_name] = {
                "matched": matched,
                "scorable": scorable,
                "match_rate": matched / scorable if scorable else 0.0,
            }
        return rates

    @staticmethod
    def _error_groups(case_scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        classification = Counter()
        difficulty = Counter()
        damage = Counter()
        trap_status = Counter()
        for case in case_scores:
            classification[str(case["classification_reason"])] += 1
            difficulty[str(case.get("difficulty") or "unspecified")] += 1
            damage[str(case.get("damage_profile") or "unspecified")] += 1
            trap_status["adversarial" if case.get("traps_present") else "non_adversarial"] += 1
        return {
            "classification_reason": dict(sorted(classification.items())),
            "difficulty": dict(sorted(difficulty.items())),
            "damage_profile": dict(sorted(damage.items())),
            "trap_status": dict(sorted(trap_status.items())),
        }

    @staticmethod
    def _case_outcomes(
        case_scores: Sequence[Mapping[str, Any]],
        golden_case_ids: Iterable[str],
        adversarial_case_ids: Iterable[str],
    ) -> dict[str, Any]:
        golden = set(golden_case_ids)
        adversarial = set(adversarial_case_ids)
        outcomes = {}
        for case in case_scores:
            case_id = str(case["case_id"])
            outcomes[case_id] = {
                "correct": case["classification_reason"] == "correct",
                "classification_reason": case["classification_reason"],
                "golden": case_id in golden
                or str(case.get("difficulty", "")).casefold() == "golden",
                "adversarial": case_id in adversarial or bool(case.get("traps_present")),
            }
        return outcomes

    def evaluate(
        self,
        *,
        truth_path: Path,
        submission_path: Path,
        split: SplitEvidence,
        run_metrics: RunMetrics | None = None,
        golden_case_ids: Iterable[str] = (),
        adversarial_case_ids: Iterable[str] = (),
        runtime_artifact_paths: Iterable[Path] = (),
        coverage: str = "holdout",
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mib-evaluation-") as temp_dir:
            aggregate_path = Path(temp_dir) / "evaluation.json"
            case_path = Path(temp_dir) / "case_scores.jsonl"
            command = [
                sys.executable,
                str(self._evaluator),
                "--truth",
                str(truth_path.resolve()),
                "--submission",
                str(submission_path.resolve()),
                "--output-json",
                str(aggregate_path),
                "--case-scores-jsonl",
                str(case_path),
            ]
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=self._repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            elapsed = time.monotonic() - started
            peak = _peak_memory_mib(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            if completed.returncode not in {0, 2} or not aggregate_path.is_file():
                raise RuntimeError(
                    "official evaluator failed: "
                    + (completed.stderr.strip() or completed.stdout.strip())
                )
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            case_scores = [
                json.loads(line)
                for line in case_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        metrics = run_metrics or RunMetrics(
            runtime_seconds=elapsed,
            peak_memory_mib=peak,
            source="official_evaluator_process_only",
        )
        counts = aggregate["counts"]
        invalid_rows = sum(
            counts[key]
            for key in (
                "duplicate_case_ids",
                "extra_cases",
                "blank_case_rows",
                "invalid_adjudication_records",
                "invalid_confidence_records",
                "invalid_fee_status_records",
            )
        )
        leakage_findings: list[str] = []
        for artifact_path in runtime_artifact_paths:
            artifact = Path(artifact_path)
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                leakage_findings.append(f"{artifact}: cannot inspect runtime artifact: {exc}")
                continue
            leakage_findings.extend(
                f"{artifact}:{finding}"
                for finding in RuntimeArtifactLeakageScanner.findings(payload)
            )

        report = {
            "report_version": "mib_evaluation_harness_v1",
            "benchmark_context": "local_engineering_benchmark_not_official_leaderboard",
            "coverage": coverage,
            "split": split.to_dict(),
            "official_evaluator": aggregate,
            "summary": {
                "total_score": aggregate["scores"]["total_score"],
                "extraction_score": aggregate["scores"]["extraction_score"],
                "classification_score": aggregate["scores"]["classification_score"],
                "calibration_score": aggregate["scores"]["calibration_score"],
                "catastrophic_false_approvals": aggregate["raw"][
                    "catastrophic_false_approvals"
                ],
                "missing_rows": counts["missing_cases"],
                "invalid_rows": invalid_rows,
            },
            "measurements": metrics.to_dict(),
            "per_field_match_rates": self._field_rates(case_scores),
            "error_groups": self._error_groups(case_scores),
            "case_outcomes": self._case_outcomes(
                case_scores, golden_case_ids, adversarial_case_ids
            ),
            "leakage_findings": leakage_findings,
        }
        return report


class InstrumentedBenchmarkRunner:
    """Run the real offline pipeline while retaining development-only traces."""

    def __init__(self, *, max_workers: int = 4) -> None:
        if not 1 <= max_workers <= 4:
            raise EvaluationConfigurationError("max_workers must be between 1 and 4")
        self._max_workers = max_workers

    def run(
        self,
        *,
        input_dir: Path,
        truth_path: Path,
        selected_case_ids: Iterable[str],
        split_name: str,
        predictions_path: Path,
        samples_path: Path,
    ) -> dict[str, Any]:
        from mib_pipeline import (
            AdjudicationEngine,
            BatchRunner,
            CaseLinker,
            ConfidenceCalibrator,
            DocumentRenderer,
            EvidencePrecedenceResolver,
            GeneralizablePolicyExceptionStore,
            VisibleEvidenceExtractor,
        )

        selected = tuple(sorted(set(selected_case_ids)))
        if not selected:
            raise EvaluationConfigurationError("benchmark split has no cases")
        selected_set = set(selected)
        truth = {
            row["case_id"]: row
            for row in read_csv(truth_path)
            if row.get("case_id") in selected_set
        }
        if set(truth) != set(selected):
            missing_truth = sorted(set(selected) - set(truth))
            raise EvaluationConfigurationError(
                f"truth labels are missing {len(missing_truth)} selected cases"
            )
        calibrator = ConfidenceCalibrator.from_pinned_artifact()
        exception_store = GeneralizablePolicyExceptionStore.from_pinned_artifact()
        engine = AdjudicationEngine(
            calibrator=calibrator,
            exceptions=exception_store,
        )
        records: dict[str, dict[str, Any]] = {}
        records_lock = threading.Lock()

        class InstrumentedProcessor:
            def __init__(self) -> None:
                self.renderer = DocumentRenderer()
                self.extractor = VisibleEvidenceExtractor()
                self.linker = CaseLinker()
                self.resolver = EvidencePrecedenceResolver()

            def process_case(self, pdf_path: Path) -> Any:
                rendered = self.renderer.render(pdf_path)
                candidates = self.extractor.extract(rendered)
                linked = self.linker.link(rendered.case_id, candidates)
                resolved = self.resolver.resolve(linked)
                outcome = engine.adjudicate_case(resolved)
                with records_lock:
                    records[outcome.row.case_id] = {
                        "case_id": outcome.row.case_id,
                        "split_name": split_name,
                        "raw_signal": calibrator.raw_signal(outcome.trace),
                        "correct": outcome.row.adjudication
                        == truth[outcome.row.case_id]["adjudication"],
                        "emitted_adjudication": outcome.row.adjudication,
                        "truth_adjudication": truth[outcome.row.case_id]["adjudication"],
                        "denial_reasons": list(outcome.trace.denial_reasons),
                        "review_reasons": list(outcome.trace.review_reasons),
                        "approval_facts": list(outcome.trace.approval_facts),
                    }
                return outcome.row

        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mib-benchmark-input-") as directory:
            selected_dir = Path(directory)
            missing_pdfs: list[str] = []
            for case_id in selected:
                source = input_dir / f"{case_id}.pdf"
                if not source.is_file():
                    missing_pdfs.append(case_id)
                    continue
                (selected_dir / source.name).symlink_to(source.resolve())
            if missing_pdfs:
                raise EvaluationConfigurationError(
                    f"input directory is missing {len(missing_pdfs)} selected PDFs"
                )
            started = time.monotonic()
            batch_report = BatchRunner(
                InstrumentedProcessor(), max_workers=self._max_workers
            ).run(selected_dir, predictions_path)
            elapsed = time.monotonic() - started
        peak = _peak_memory_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        ordered_records = [records[case_id] for case_id in sorted(records)]
        samples_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in ordered_records),
            encoding="utf-8",
        )
        return {
            "split_name": split_name,
            "attempted": batch_report.attempted,
            "answered": batch_report.answered,
            "omitted": batch_report.omitted,
            "runtime_seconds": elapsed,
            "peak_memory_mib": peak,
            "calibration_artifact_id": calibrator.artifact_id,
            "policy_exception_artifact_id": exception_store.artifact_id,
            "failures": [
                {"source_name": failure.source_name, "reason": failure.reason}
                for failure in batch_report.failures
            ],
        }


class ReleaseGate:
    """Block unsafe or dishonestly measured prediction-logic changes."""

    SCORE_KEYS = (
        "total_score",
        "extraction_score",
        "classification_score",
        "calibration_score",
    )

    @staticmethod
    def _regressions(
        candidate: Mapping[str, Any],
        baseline: Mapping[str, Any],
        category: str,
    ) -> list[str]:
        candidate_cases = candidate.get("case_outcomes", {})
        baseline_cases = baseline.get("case_outcomes", {})
        return sorted(
            case_id
            for case_id, before in baseline_cases.items()
            if before.get(category)
            and before.get("correct")
            and case_id in candidate_cases
            and not candidate_cases[case_id].get("correct")
        )

    def decide(
        self,
        *,
        candidate: Mapping[str, Any],
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        blocking: list[str] = []
        warnings: list[str] = []
        candidate_split = candidate.get("split", {})
        baseline_split = baseline.get("split", {})
        if not candidate_split.get("is_honest_holdout"):
            blocking.append("candidate result is not from an honest non-tuning split")
        if not baseline_split.get("is_honest_holdout"):
            blocking.append("baseline result is not from an honest non-tuning split")
        if candidate_split.get("name") != baseline_split.get("name"):
            blocking.append("candidate and baseline were not measured on the same split")
        if set(candidate.get("case_outcomes", {})) != set(
            baseline.get("case_outcomes", {})
        ):
            blocking.append("candidate and baseline do not cover the same cases")
        measurements = candidate.get("measurements", {})
        for key in ("runtime_seconds", "peak_memory_mib", "source"):
            if key not in measurements:
                blocking.append(f"required measurement is missing: {key}")
        if measurements.get("runtime_seconds", 0) <= 0:
            blocking.append("offline runtime duration was not measured")
        if measurements.get("peak_memory_mib", 0) <= 0:
            blocking.append("offline runtime peak memory was not measured")
        if measurements.get("source") == "official_evaluator_process_only":
            blocking.append("only evaluator overhead was measured, not the offline runtime")
        if candidate.get("leakage_findings"):
            blocking.append("case-identity leakage was detected")

        candidate_summary = candidate.get("summary", {})
        baseline_summary = baseline.get("summary", {})
        required_summary = self.SCORE_KEYS + (
            "catastrophic_false_approvals",
            "missing_rows",
            "invalid_rows",
        )
        for key in required_summary:
            if key not in candidate_summary:
                blocking.append(f"required evaluation result is missing: {key}")

        before_false_approvals = baseline_summary.get("catastrophic_false_approvals", 0)
        after_false_approvals = candidate_summary.get(
            "catastrophic_false_approvals", before_false_approvals + 1
        )
        if after_false_approvals > before_false_approvals:
            blocking.append("catastrophic false approvals increased")

        golden_regressions = self._regressions(candidate, baseline, "golden")
        adversarial_regressions = self._regressions(candidate, baseline, "adversarial")
        if golden_regressions:
            warnings.append("new golden-case regressions")
        if adversarial_regressions:
            warnings.append("new adversarial-case regressions")
        if candidate_summary.get("invalid_rows", 0) > baseline_summary.get(
            "invalid_rows", 0
        ):
            warnings.append("invalid row count increased")
        if candidate_summary.get("missing_rows", 0) > baseline_summary.get(
            "missing_rows", 0
        ):
            warnings.append("missing row count increased")

        score_deltas = {
            key: candidate_summary.get(key, 0) - baseline_summary.get(key, 0)
            for key in self.SCORE_KEYS
        }
        if blocking:
            decision = "BLOCKED"
        elif warnings:
            decision = "PASS_WITH_WARNINGS"
        else:
            decision = "PASS"
        return {
            "gate_version": "mib_false_approval_gate_v1",
            "decision": decision,
            "adopt": not blocking,
            "blocking_reasons": blocking,
            "warnings": warnings,
            "score_deltas": score_deltas,
            "catastrophic_false_approval_delta": after_false_approvals
            - before_false_approvals,
            "golden_regressions": golden_regressions,
            "adversarial_regressions": adversarial_regressions,
        }


class IsotonicCalibrationFitter:
    """Fit a deterministic PAV isotonic map on one honest calibration split."""

    @staticmethod
    def fit(
        samples: Sequence[Mapping[str, Any]],
        *,
        artifact_id: str,
        split: SplitEvidence,
        calibration_case_ids: Iterable[str],
        tuning_case_ids: Iterable[str],
    ) -> dict[str, Any]:
        if not samples:
            raise EvaluationConfigurationError("calibration requires samples")
        if split.role != "calibration" or not split.is_honest_holdout:
            raise EvaluationConfigurationError(
                "calibration must be fit on a declared honest calibration split"
            )
        seen: set[str] = set()
        allowed = set(calibration_case_ids)
        tuning = set(tuning_case_ids)
        HoldoutSplitManager.assert_no_overlap(allowed, tuning)
        if not allowed:
            raise EvaluationConfigurationError(
                "calibration split manifest contains no cases"
            )
        grouped: dict[float, list[int]] = defaultdict(list)
        for sample in samples:
            case_id = str(sample.get("case_id", "")).strip()
            sample_split = str(sample.get("split_name", "")).strip()
            if not case_id or case_id in seen:
                raise EvaluationConfigurationError(
                    "calibration samples need unique non-empty case IDs"
                )
            if sample_split != split.name:
                raise EvaluationConfigurationError(
                    "calibration sample came from the wrong split"
                )
            if case_id not in allowed:
                raise EvaluationConfigurationError(
                    "calibration sample is absent from the calibration manifest"
                )
            seen.add(case_id)
            raw_signal = float(sample["raw_signal"])
            correct = sample["correct"]
            if not math.isfinite(raw_signal) or not 0 <= raw_signal <= 1:
                raise EvaluationConfigurationError("raw signals must be in [0,1]")
            if not isinstance(correct, bool):
                raise EvaluationConfigurationError("calibration targets must be booleans")
            grouped[raw_signal].append(int(correct))

        blocks: list[dict[str, float]] = []
        for raw_signal in sorted(grouped):
            targets = grouped[raw_signal]
            blocks.append(
                {
                    "start": raw_signal,
                    "end": raw_signal,
                    "sum": float(sum(targets)),
                    "count": float(len(targets)),
                }
            )
            while len(blocks) >= 2:
                previous = blocks[-2]
                current = blocks[-1]
                previous_mean = previous["sum"] / previous["count"]
                current_mean = current["sum"] / current["count"]
                if previous_mean <= current_mean:
                    break
                blocks[-2:] = [
                    {
                        "start": previous["start"],
                        "end": current["end"],
                        "sum": previous["sum"] + current["sum"],
                        "count": previous["count"] + current["count"],
                    }
                ]

        breakpoints: list[float] = []
        probabilities: list[float] = []
        first_probability = blocks[0]["sum"] / blocks[0]["count"]
        if blocks[0]["start"] > 0:
            breakpoints.append(0.0)
            probabilities.append(first_probability)
        for block in blocks:
            breakpoints.append(block["start"])
            probabilities.append(block["sum"] / block["count"])
        if breakpoints[-1] < 1.0:
            breakpoints.append(1.0)
            probabilities.append(probabilities[-1])

        artifact = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "breakpoints": breakpoints,
            "probabilities": probabilities,
            "fit_metadata": {
                "method": "pool_adjacent_violators_isotonic_regression",
                "fit_split": split.name,
                "fit_split_role": split.role,
                "training_case_count": len(samples),
                "positive_target_count": sum(int(sample["correct"]) for sample in samples),
                "target": "emitted_adjudication_is_correct",
            },
        }
        RuntimeArtifactLeakageScanner.require_clean(artifact)
        return artifact


class PolicyExceptionValidator:
    """Admit only visible, held-out-supported, safety-neutral strict exceptions."""

    def __init__(self, *, minimum_heldout_support: int = 2) -> None:
        self._minimum_heldout_support = minimum_heldout_support

    def validate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        evidence: Mapping[str, Mapping[str, Any]],
        *,
        artifact_id: str,
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        # Import here so the development-only harness is not a runtime dependency.
        from mib_pipeline.adjudication import (
            GeneralizablePolicyExceptionStore,
            PolicyException,
        )

        accepted: list[dict[str, Any]] = []
        rejected: dict[str, list[str]] = {}
        for raw_rule in candidates:
            rule_id = str(raw_rule.get("rule_id", ""))
            reasons: list[str] = []
            rule = PolicyException(
                rule_id=rule_id,
                conditions=raw_rule.get("conditions", {}),
                decision=str(raw_rule.get("decision", "")),
                rationale=str(raw_rule.get("rationale", "")),
            )
            try:
                GeneralizablePolicyExceptionStore([rule])
            except ValueError as exc:
                reasons.append(str(exc))
            proof = evidence.get(rule_id, {})
            visible_support = set(proof.get("visible_support_case_ids", []))
            heldout_support = set(proof.get("heldout_support_case_ids", []))
            tuning_cases = set(proof.get("tuning_case_ids", []))
            if not visible_support:
                reasons.append("no trusted visible support")
            if len(heldout_support) < self._minimum_heldout_support:
                reasons.append("insufficient held-out support")
            if not heldout_support.issubset(visible_support):
                reasons.append("held-out support lacks trusted visible evidence")
            if heldout_support & tuning_cases:
                reasons.append("held-out support overlaps tuning cases")
            if proof.get("tuning_split") == proof.get("measured_split"):
                reasons.append("exception was measured on its tuning split")
            if int(proof.get("candidate_false_approvals", 0)) > int(
                proof.get("baseline_false_approvals", 0)
            ):
                reasons.append("exception increases catastrophic false approvals")
            if reasons:
                rejected[rule_id or "<missing>"] = reasons
                continue
            accepted.append(
                {
                    "rule_id": rule.rule_id,
                    "conditions": dict(rule.conditions),
                    "decision": rule.decision,
                    "rationale": rule.rationale,
                }
            )
        artifact = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "exceptions": sorted(accepted, key=lambda rule: rule["rule_id"]),
        }
        RuntimeArtifactLeakageScanner.require_clean(artifact)
        return artifact, rejected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise EvaluationConfigurationError("cannot write an empty split label file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
