import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solution.py"

sys.path.insert(0, str(ROOT))
import solution  # noqa: E402


class RuntimeScaffoldTests(unittest.TestCase):
    def run_solution(self, *arguments, env=None):
        return subprocess.run(
            [sys.executable, str(SOLUTION), *map(str, arguments)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_requires_exactly_two_arguments(self):
        for arguments in ((), ("/input",), ("/input", "/output/p.jsonl", "extra")):
            with self.subTest(arguments=arguments):
                result = self.run_solution(*arguments)
                self.assertEqual(result.returncode, 64)
                self.assertIn("error:", result.stderr)

    def test_empty_input_creates_empty_canonical_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            output_path = output_dir / "predictions.jsonl"

            result = self.run_solution(input_dir, output_path)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output_path.read_bytes(), b"")

    def test_case_discovery_is_pdf_only_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            for name in ("case-b.pdf", "A-case.PDF", "notes.txt", "case-a.pdf"):
                (input_dir / name).touch()
            (input_dir / "nested").mkdir()
            (input_dir / "nested" / "ignored.pdf").touch()

            discovered = solution.discover_case_pdfs(input_dir)

            self.assertEqual(
                [path.name for path in discovered],
                ["A-case.PDF", "case-a.pdf", "case-b.pdf"],
            )

    def test_output_must_not_be_written_inside_read_only_input(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir) / "input"
            input_dir.mkdir()
            output_path = input_dir / "predictions.jsonl"

            result = self.run_solution(input_dir, output_path)

            self.assertEqual(result.returncode, 64)
            self.assertFalse(output_path.exists())
            self.assertIn("must not be inside", result.stderr)

    def test_output_parent_is_not_created_outside_the_supplied_mount(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()
            missing_output_dir = root / "missing" / "nested"

            result = self.run_solution(
                input_dir, missing_output_dir / "predictions.jsonl"
            )

            self.assertEqual(result.returncode, 64)
            self.assertFalse(missing_output_dir.exists())

    def test_worker_limit_is_bounded_to_four(self):
        original = os.environ.get("MIB_MAX_WORKERS")
        try:
            os.environ["MIB_MAX_WORKERS"] = "999"
            self.assertEqual(solution.configured_worker_limit(), 4)
            os.environ["MIB_MAX_WORKERS"] = "2"
            self.assertEqual(solution.configured_worker_limit(), 2)
        finally:
            if original is None:
                os.environ.pop("MIB_MAX_WORKERS", None)
            else:
                os.environ["MIB_MAX_WORKERS"] = original

    def test_shell_and_docker_descriptors_match_the_contract(self):
        run_script = (ROOT / "run.sh").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()

        self.assertIn('if [ "$#" -ne 2 ]', run_script)
        self.assertIn('ENTRYPOINT ["/app/run.sh"]', dockerfile)
        self.assertIn("FROM python:3.12.11-slim-bookworm", dockerfile)
        self.assertIn("USER mib:mib", dockerfile)
        self.assertIn("MIB_MAX_WORKERS=4", dockerfile)


if __name__ == "__main__":
    unittest.main()
