import json
import tempfile
import unittest
from pathlib import Path

from mib_pipeline import (
    BatchRunner,
    CanonicalJsonlWriter,
    DuplicateCaseIdError,
    FIELD_NAMES,
    PredictionRow,
    SafeFallbackProcessor,
)


def valid_mapping(case_id="MIB-000001"):
    return {
        "case_id": case_id,
        "applicant_name": "Zed",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1042",
        "arrival_date": "2026-04-17",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "APPROVED",
        "confidence": 0.9,
    }


class CanonicalWriterTests(unittest.TestCase):
    def test_writer_emits_exact_fields_one_object_per_line(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "predictions.jsonl"
            row = {**valid_mapping(), "ignored_extra": "not emitted"}

            CanonicalJsonlWriter().write(output, [row])

            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(tuple(parsed), FIELD_NAMES)
            self.assertNotIn("ignored_extra", parsed)

    def test_writer_sorts_rows_for_byte_determinism(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            rows = [valid_mapping("MIB-000002"), valid_mapping("MIB-000001")]

            writer = CanonicalJsonlWriter()
            writer.write(first, rows)
            writer.write(second, reversed(rows))

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                [json.loads(line)["case_id"] for line in first.read_text().splitlines()],
                ["MIB-000001", "MIB-000002"],
            )

    def test_invalid_computed_values_become_safe_schema_values(self):
        row = PredictionRow.from_mapping(
            {
                "case_id": "bad",
                "sponsor_id": "bad",
                "arrival_date": "tomorrow",
                "fee_status": "maybe",
                "adjudication": "YES",
                "confidence": float("nan"),
            },
            fallback_case_id="MIB-000001",
        )

        self.assertEqual(row.case_id, "MIB-000001")
        self.assertEqual(row.sponsor_id, "SPN-0000")
        self.assertEqual(row.arrival_date, "1900-01-01")
        self.assertEqual(row.fee_status, "unknown")
        self.assertEqual(row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(row.confidence, 0.0)

    def test_duplicate_ids_are_rejected_before_existing_output_is_replaced(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "predictions.jsonl"
            output.write_text("existing\n")

            with self.assertRaises(DuplicateCaseIdError):
                CanonicalJsonlWriter().write(
                    output,
                    [valid_mapping(), valid_mapping()],
                )

            self.assertEqual(output.read_text(), "existing\n")

    def test_empty_rows_create_empty_valid_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output = Path(temporary_dir) / "predictions.jsonl"
            CanonicalJsonlWriter().write(output, [])
            self.assertEqual(output.read_bytes(), b"")


class BatchRunnerTests(unittest.TestCase):
    def test_runner_attempts_every_pdf_and_isolates_case_failures(self):
        class SelectiveProcessor:
            def process_case(self, pdf_path):
                if pdf_path.name == "MIB-000002.pdf":
                    raise RuntimeError("corrupt PDF")
                return valid_mapping(pdf_path.stem)

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            output = root / "predictions.jsonl"
            for case_id in ("MIB-000001", "MIB-000002", "MIB-000003"):
                (input_dir / f"{case_id}.pdf").touch()

            report = BatchRunner(SelectiveProcessor(), max_workers=3).run(
                input_dir, output
            )

            self.assertEqual(report.attempted, 3)
            self.assertEqual(report.answered, 2)
            self.assertEqual(report.omitted, 1)
            self.assertEqual(report.failures[0].source_name, "MIB-000002.pdf")
            self.assertEqual(
                [json.loads(line)["case_id"] for line in output.read_text().splitlines()],
                ["MIB-000001", "MIB-000003"],
            )

    def test_fallback_processor_answers_large_batch_without_interaction(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            output = root / "predictions.jsonl"
            for index in range(1, 201):
                (input_dir / f"MIB-{index:06d}.pdf").touch()

            report = BatchRunner(SafeFallbackProcessor()).run(input_dir, output)

            self.assertEqual(report.attempted, 200)
            self.assertEqual(report.answered, 200)
            self.assertEqual(report.omitted, 0)
            self.assertEqual(len(output.read_text().splitlines()), 200)

    def test_case_without_recoverable_id_is_omitted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "unknown.pdf").touch()
            output = root / "predictions.jsonl"

            report = BatchRunner(SafeFallbackProcessor()).run(input_dir, output)

            self.assertEqual(report.attempted, 1)
            self.assertEqual(report.answered, 0)
            self.assertEqual(report.omitted, 1)
            self.assertEqual(output.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
