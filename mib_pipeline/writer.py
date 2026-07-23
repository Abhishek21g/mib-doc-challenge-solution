"""Atomic canonical JSONL serialization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Any

from .models import PredictionRow


class DuplicateCaseIdError(ValueError):
    """Raised before writing when two rows share a case ID."""


class CanonicalJsonlWriter:
    """Write deterministic, exact-shape prediction rows atomically."""

    def write(
        self,
        output_path: Path,
        rows: Iterable[PredictionRow | Mapping[str, Any]],
    ) -> None:
        normalized_rows = tuple(
            row if isinstance(row, PredictionRow) else PredictionRow.from_mapping(row)
            for row in rows
        )
        case_ids = [row.case_id for row in normalized_rows]
        if len(case_ids) != len(set(case_ids)):
            duplicates = sorted(
                case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
            )
            raise DuplicateCaseIdError(
                f"duplicate case_id values: {', '.join(duplicates)}"
            )

        ordered_rows = sorted(normalized_rows, key=lambda row: row.case_id)
        output_path = Path(output_path)
        if not output_path.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {output_path.parent}")

        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                descriptor = -1
                for row in ordered_rows:
                    payload = json.dumps(
                        row.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    handle.write(payload)
                    handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary_path), str(output_path))
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise
