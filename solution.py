#!/usr/bin/env python3
"""Offline two-argument runtime for the MIB case processor."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from mib_pipeline import (
    AdjudicationEngine,
    BatchRunner,
    CalibrationArtifactError,
    CaseLinker,
    ConfidenceCalibrator,
    DocumentRenderer,
    EvidencePrecedenceResolver,
    GeneralizablePolicyExceptionStore,
    OutputConfidenceRecalibrationProcessor,
    OutputConfidenceRecalibrator,
    PolicyArtifactError,
    RapidOutputRecoveryProcessor,
    ReviewDenialRecoveryAdjudicator,
    VisibleEvidenceExtractor,
    discover_case_pdfs,
)


USAGE = "usage: solution.py <input_pdf_dir> <output_predictions_path>"
MAX_WORKERS = 4


class ContractError(ValueError):
    """Raised when the two-argument runtime contract is not satisfied."""


def configured_worker_limit() -> int:
    """Return a valid worker limit that never exceeds the four-vCPU budget."""

    raw_value = os.environ.get("MIB_MAX_WORKERS", str(MAX_WORKERS))
    try:
        requested = int(raw_value)
    except ValueError as exc:
        raise ContractError("MIB_MAX_WORKERS must be an integer") from exc
    if requested < 1:
        raise ContractError("MIB_MAX_WORKERS must be at least 1")
    return min(requested, MAX_WORKERS)


def parse_paths(argv: Sequence[str]) -> tuple[Path, Path]:
    """Validate and return the input directory and exact output file path."""

    if len(argv) != 3:
        raise ContractError(USAGE)

    input_dir = Path(argv[1])
    output_path = Path(argv[2])

    if not input_dir.is_dir():
        raise ContractError(f"input PDF directory does not exist: {input_dir}")
    if not output_path.name:
        raise ContractError("output predictions path must name a file")
    if output_path.exists() and output_path.is_dir():
        raise ContractError(f"output predictions path is a directory: {output_path}")
    if not output_path.parent.is_dir():
        raise ContractError(
            f"output directory does not exist: {output_path.parent}"
        )

    resolved_input = input_dir.resolve()
    resolved_output = output_path.resolve()
    try:
        resolved_output.relative_to(resolved_input)
    except ValueError:
        output_is_inside_input = False
    else:
        output_is_inside_input = True
    if output_is_inside_input:
        raise ContractError("output predictions path must not be inside the input directory")

    return input_dir, output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline processor and return a process exit code."""

    arguments = sys.argv if argv is None else argv
    try:
        input_dir, output_path = parse_paths(arguments)
        runner = BatchRunner(
            OutputConfidenceRecalibrationProcessor(
                processor=RapidOutputRecoveryProcessor(
                    renderer=DocumentRenderer(),
                    primary_extractor=VisibleEvidenceExtractor(
                        packet_page_type_markers=True,
                    ),
                    linker=CaseLinker(),
                    resolver=EvidencePrecedenceResolver(),
                    adjudicator=ReviewDenialRecoveryAdjudicator(
                        AdjudicationEngine(
                            calibrator=ConfidenceCalibrator.from_pinned_artifact(),
                            exceptions=GeneralizablePolicyExceptionStore.from_pinned_artifact(),
                        )
                    ),
                ),
                recalibrator=OutputConfidenceRecalibrator.from_pinned_artifact(),
            ),
            max_workers=configured_worker_limit(),
        )
        report = runner.run(input_dir, output_path)
    except (CalibrationArtifactError, ContractError, OSError, PolicyArtifactError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 64
    print(
        f"attempted={report.attempted} answered={report.answered} omitted={report.omitted}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
