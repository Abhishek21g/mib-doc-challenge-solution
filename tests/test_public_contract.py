import csv
import json
import unittest
from pathlib import Path

from scripts import evaluate
from scripts import validate_submission


ROOT = Path(__file__).resolve().parents[1]


def example_prediction():
    return json.loads((ROOT / "examples" / "submission.jsonl").read_text().splitlines()[0])


class SubmissionValidationTests(unittest.TestCase):
    def test_jsonl_examples_match_the_public_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed")

        schema = json.loads((ROOT / "schemas" / "submission.schema.json").read_text())
        rows, submission_format = validate_submission.read_submission(
            ROOT / "examples" / "submission.jsonl"
        )

        for index, row in enumerate(rows, start=1):
            self.assertEqual(
                validate_submission.validate_row(row, f"record {index}", submission_format),
                [],
            )
            jsonschema.validate(row, schema, format_checker=jsonschema.FormatChecker())

    def test_csv_and_jsonl_examples_are_equivalent(self):
        csv_rows, csv_format = validate_submission.read_submission(
            ROOT / "examples" / "submission.csv"
        )
        json_rows, _ = validate_submission.read_submission(
            ROOT / "examples" / "submission.jsonl"
        )

        normalized_json_rows = [
            {key: str(value) if key == "confidence" else value for key, value in row.items()}
            for row in json_rows
        ]
        self.assertEqual(csv_format, "csv")
        self.assertEqual(csv_rows, normalized_json_rows)

    def test_validator_rejects_values_rejected_by_schema(self):
        invalid_rows = [
            {**example_prediction(), "case_id": "MIB-1"},
            {**example_prediction(), "sponsor_id": "SPN-12"},
            {**example_prediction(), "arrival_date": "2026-99-99"},
            {**example_prediction(), "adjudication": "approved"},
            {**example_prediction(), "applicant_name": 123},
            {**example_prediction(), "confidence": True},
            {**example_prediction(), "confidence": "0.9"},
        ]

        for index, row in enumerate(invalid_rows, start=1):
            with self.subTest(index=index):
                self.assertTrue(validate_submission.validate_row(row, "record", "json"))


class EvaluationTests(unittest.TestCase):
    def test_perfect_training_predictions_score_150(self):
        truth = evaluate.read_truth(ROOT / "data" / "train_labels.csv")
        predictions = []
        for row in truth.values():
            prediction = dict(row)
            prediction["confidence"] = 1.0
            predictions.append(prediction)

        results, case_scores = evaluate.build_results(truth, predictions)

        self.assertEqual(len(case_scores), 1000)
        self.assertAlmostEqual(results["scores"]["total_score"], 150.0)

    def test_evaluation_result_matches_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema is not installed")

        with (ROOT / "data" / "train_labels.csv").open(newline="") as handle:
            truth_row = next(csv.DictReader(handle))
        prediction = dict(truth_row)
        prediction["confidence"] = 1.0
        results, _ = evaluate.build_results({truth_row["case_id"]: truth_row}, [prediction])
        schema = json.loads((ROOT / "schemas" / "evaluation-result.schema.json").read_text())

        jsonschema.validate(results, schema)


if __name__ == "__main__":
    unittest.main()
