#!/usr/bin/env python3
"""Development CLI for honest evaluation, calibration, and release gating."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.evaluation import (  # noqa: E402
    EvaluationHarness,
    HoldoutSplitManager,
    IsotonicCalibrationFitter,
    InstrumentedBenchmarkRunner,
    PolicyExceptionValidator,
    ReleaseGate,
    RunMetrics,
    SplitEvidence,
    read_csv,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
)


RUNTIME_ARTIFACT_DIR = REPO_ROOT / "mib_pipeline" / "artifacts"


def _outside_runtime_artifacts(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(RUNTIME_ARTIFACT_DIR.resolve())
    except ValueError:
        return resolved
    raise SystemExit(
        "development reports and split manifests must not be written into runtime artifacts"
    )


def _manifest_ids(path: str | None) -> tuple[str, ...]:
    if not path:
        return ()
    return tuple(
        str(row.get("case_id", "")).strip()
        for row in read_csv(Path(path))
        if str(row.get("case_id", "")).strip()
    )


def command_split(args: argparse.Namespace) -> int:
    manager = HoldoutSplitManager(
        seed=args.seed,
        tuning_fraction=args.tuning_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    truth_rows = read_csv(Path(args.truth))
    forced_tuning_case_ids = _manifest_ids(args.forced_tuning_manifest)
    splits = manager.split_rows(
        truth_rows,
        forced_tuning_case_ids=forced_tuning_case_ids,
    )
    payload = {
        "manifest_version": "mib_honest_holdout_v1",
        "seed_label": args.seed_label,
        "forced_tuning_case_count": len(forced_tuning_case_ids),
        "splits": {name: list(case_ids) for name, case_ids in splits.items()},
        "counts": {name: len(case_ids) for name, case_ids in splits.items()},
    }
    write_json(_outside_runtime_artifacts(Path(args.output)), payload)
    if args.label_output_dir:
        output_dir = _outside_runtime_artifacts(Path(args.label_output_dir))
        for role, case_ids in splits.items():
            selected = set(case_ids)
            write_csv(
                output_dir / f"{role}.csv",
                [row for row in truth_rows if row["case_id"] in selected],
            )
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    metrics = None
    provided = args.runtime_seconds is not None or args.peak_memory_mib is not None
    if provided:
        if args.runtime_seconds is None or args.peak_memory_mib is None:
            raise SystemExit("provide both --runtime-seconds and --peak-memory-mib")
        metrics = RunMetrics(
            runtime_seconds=args.runtime_seconds,
            peak_memory_mib=args.peak_memory_mib,
            source=args.metrics_source,
        )
    split = SplitEvidence(
        name=args.split_name,
        role=args.split_role,
        tuned_on_splits=tuple(args.tuned_on),
    )
    report = EvaluationHarness(REPO_ROOT).evaluate(
        truth_path=Path(args.truth),
        submission_path=Path(args.submission),
        split=split,
        run_metrics=metrics,
        golden_case_ids=_manifest_ids(args.golden_manifest),
        adversarial_case_ids=_manifest_ids(args.adversarial_manifest),
        runtime_artifact_paths=(
            tuple(Path(path) for path in args.runtime_artifact)
            if args.runtime_artifact
            else (
                RUNTIME_ARTIFACT_DIR / "confidence_calibration.json",
                RUNTIME_ARTIFACT_DIR / "policy_exceptions.json",
            )
        ),
        coverage=args.coverage,
    )
    write_json(_outside_runtime_artifacts(Path(args.output)), report)
    summary = report["summary"]
    print(
        "local engineering benchmark: "
        f"total={summary['total_score']:.2f} "
        f"extraction={summary['extraction_score']:.2f} "
        f"classification={summary['classification_score']:.2f} "
        f"calibration={summary['calibration_score']:.2f} "
        f"false_approvals={summary['catastrophic_false_approvals']}"
    )
    return 0


def command_gate(args: argparse.Namespace) -> int:
    decision = ReleaseGate().decide(
        candidate=read_json(Path(args.candidate_report)),
        baseline=read_json(Path(args.baseline_report)),
    )
    write_json(_outside_runtime_artifacts(Path(args.output)), decision)
    print(f"release gate: {decision['decision']}")
    for reason in decision["blocking_reasons"]:
        print(f"BLOCK: {reason}")
    for warning in decision["warnings"]:
        print(f"WARN: {warning}")
    return 0 if decision["adopt"] else 3


def command_run_benchmark(args: argparse.Namespace) -> int:
    truth_rows = read_csv(Path(args.truth))
    if args.split_role == "full_training":
        selected_case_ids = [row["case_id"] for row in truth_rows]
    else:
        if not args.split_manifest:
            raise SystemExit("--split-manifest is required outside full_training")
        manifest = read_json(Path(args.split_manifest))
        selected_case_ids = manifest.get("splits", {}).get(args.split_role, [])
    report = InstrumentedBenchmarkRunner(max_workers=args.max_workers).run(
        input_dir=Path(args.input_dir),
        truth_path=Path(args.truth),
        selected_case_ids=selected_case_ids,
        split_name=args.split_name,
        predictions_path=Path(args.predictions),
        samples_path=Path(args.samples),
    )
    write_json(_outside_runtime_artifacts(Path(args.run_report)), report)
    print(
        f"benchmark attempted={report['attempted']} answered={report['answered']} "
        f"omitted={report['omitted']} runtime={report['runtime_seconds']:.2f}s "
        f"peak_memory={report['peak_memory_mib']:.1f}MiB"
    )
    return 0


def command_fit_calibration(args: argparse.Namespace) -> int:
    split = SplitEvidence(
        name=args.split_name,
        role="calibration",
        tuned_on_splits=tuple(args.tuned_on),
    )
    manifest = read_json(Path(args.split_manifest))
    manifest_splits = manifest.get("splits", {})
    artifact = IsotonicCalibrationFitter.fit(
        read_jsonl(Path(args.samples)),
        artifact_id=args.artifact_id,
        split=split,
        calibration_case_ids=manifest_splits.get("calibration", ()),
        tuning_case_ids=manifest_splits.get("tuning", ()),
    )
    write_json(Path(args.output), artifact)
    print(
        f"fitted {artifact['artifact_id']} on "
        f"{artifact['fit_metadata']['training_case_count']} honest holdout cases"
    )
    return 0


def command_validate_exceptions(args: argparse.Namespace) -> int:
    candidate_payload = read_json(Path(args.candidates))
    candidates = (
        candidate_payload.get("exceptions", [])
        if isinstance(candidate_payload, dict)
        else candidate_payload
    )
    artifact, rejected = PolicyExceptionValidator(
        minimum_heldout_support=args.minimum_heldout_support
    ).validate(
        candidates,
        read_json(Path(args.evidence)),
        artifact_id=args.artifact_id,
    )
    write_json(Path(args.output), artifact)
    write_json(
        _outside_runtime_artifacts(Path(args.report)),
        {
            "accepted": [rule["rule_id"] for rule in artifact["exceptions"]],
            "rejected": rejected,
        },
    )
    print(f"validated exceptions: {len(artifact['exceptions'])}")
    print(f"rejected exceptions: {len(rejected)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MIB development-only evaluation and release-gate harness"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="create deterministic honest splits")
    split.add_argument("--truth", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--seed", required=True)
    split.add_argument("--seed-label", default="local-reproducible-v1")
    split.add_argument("--label-output-dir")
    split.add_argument(
        "--forced-tuning-manifest",
        help="CSV of previously inspected cases that must remain in tuning",
    )
    split.add_argument("--tuning-fraction", type=float, default=0.7)
    split.add_argument("--calibration-fraction", type=float, default=0.15)
    split.set_defaults(handler=command_split)

    evaluate = commands.add_parser("evaluate", help="wrap the official evaluator")
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--submission", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split-name", required=True)
    evaluate.add_argument(
        "--split-role",
        required=True,
        choices=("tuning", "calibration", "release", "full_training", "golden", "adversarial"),
    )
    evaluate.add_argument("--tuned-on", action="append", default=[])
    evaluate.add_argument("--coverage", default="holdout")
    evaluate.add_argument("--runtime-seconds", type=float)
    evaluate.add_argument("--peak-memory-mib", type=float)
    evaluate.add_argument("--metrics-source", default="offline_runtime_measurement")
    evaluate.add_argument("--golden-manifest")
    evaluate.add_argument("--adversarial-manifest")
    evaluate.add_argument("--runtime-artifact", action="append", default=[])
    evaluate.set_defaults(handler=command_evaluate)

    gate = commands.add_parser("gate", help="enforce the false-approval gate")
    gate.add_argument("--candidate-report", required=True)
    gate.add_argument("--baseline-report", required=True)
    gate.add_argument("--output", required=True)
    gate.set_defaults(handler=command_gate)

    benchmark = commands.add_parser(
        "run-benchmark", help="run the real pipeline with development-only traces"
    )
    benchmark.add_argument("--input-dir", required=True)
    benchmark.add_argument("--truth", required=True)
    benchmark.add_argument("--split-manifest")
    benchmark.add_argument(
        "--split-role",
        required=True,
        choices=("tuning", "calibration", "release", "full_training"),
    )
    benchmark.add_argument("--split-name", required=True)
    benchmark.add_argument("--predictions", required=True)
    benchmark.add_argument("--samples", required=True)
    benchmark.add_argument("--run-report", required=True)
    benchmark.add_argument("--max-workers", type=int, default=4)
    benchmark.set_defaults(handler=command_run_benchmark)

    calibration = commands.add_parser(
        "fit-calibration", help="fit and publish an honest isotonic map"
    )
    calibration.add_argument("--samples", required=True)
    calibration.add_argument("--output", required=True)
    calibration.add_argument("--artifact-id", required=True)
    calibration.add_argument("--split-name", required=True)
    calibration.add_argument("--split-manifest", required=True)
    calibration.add_argument("--tuned-on", action="append", default=[])
    calibration.set_defaults(handler=command_fit_calibration)

    exceptions = commands.add_parser(
        "validate-exceptions", help="publish held-out-validated strict exceptions"
    )
    exceptions.add_argument("--candidates", required=True)
    exceptions.add_argument("--evidence", required=True)
    exceptions.add_argument("--output", required=True)
    exceptions.add_argument("--report", required=True)
    exceptions.add_argument("--artifact-id", required=True)
    exceptions.add_argument("--minimum-heldout-support", type=int, default=2)
    exceptions.set_defaults(handler=command_validate_exceptions)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
