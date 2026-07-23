"""Independent, bounded batch orchestration for PDF case files."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import PredictionRow
from .pipeline import CaseProcessor
from .writer import CanonicalJsonlWriter


def discover_case_pdfs(input_dir: Path) -> tuple[Path, ...]:
    """Enumerate top-level PDF cases in deterministic filename order."""

    return tuple(
        sorted(
            (
                path
                for path in Path(input_dir).iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )


@dataclass(frozen=True)
class CaseFailure:
    source_name: str
    reason: str


@dataclass(frozen=True)
class CaseResult:
    row: PredictionRow | None
    failure: CaseFailure | None


@dataclass(frozen=True)
class BatchRunReport:
    attempted: int
    answered: int
    omitted: int
    failures: tuple[CaseFailure, ...]


class BatchRunner:
    """Process every PDF independently and serialize only valid results."""

    def __init__(
        self,
        processor: CaseProcessor,
        *,
        writer: CanonicalJsonlWriter | None = None,
        max_workers: int = 4,
    ) -> None:
        if max_workers < 1 or max_workers > 4:
            raise ValueError("max_workers must be between 1 and 4")
        self._processor = processor
        self._writer = writer or CanonicalJsonlWriter()
        self._max_workers = max_workers

    def _process_one(self, pdf_path: Path) -> CaseResult:
        try:
            computed = self._processor.process_case(pdf_path)
            if computed is None:
                return CaseResult(
                    row=None,
                    failure=CaseFailure(pdf_path.name, "processor omitted case"),
                )
            row = (
                computed
                if isinstance(computed, PredictionRow)
                else PredictionRow.from_mapping(
                    computed,
                    fallback_case_id=pdf_path.stem,
                )
            )
            return CaseResult(row=row, failure=None)
        except Exception as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            return CaseResult(
                row=None,
                failure=CaseFailure(pdf_path.name, reason),
            )

    def _process_all(self, pdf_paths: Iterable[Path]) -> tuple[CaseResult, ...]:
        paths = tuple(pdf_paths)
        if len(paths) <= 1 or self._max_workers == 1:
            return tuple(self._process_one(path) for path in paths)
        with ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="mib-case",
        ) as executor:
            return tuple(executor.map(self._process_one, paths))

    def run(self, input_dir: Path, output_path: Path) -> BatchRunReport:
        pdf_paths = discover_case_pdfs(input_dir)
        results = self._process_all(pdf_paths)
        rows = tuple(result.row for result in results if result.row is not None)
        failures = tuple(
            result.failure for result in results if result.failure is not None
        )
        self._writer.write(output_path, rows)
        return BatchRunReport(
            attempted=len(pdf_paths),
            answered=len(rows),
            omitted=len(failures),
            failures=failures,
        )
