from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .adjudicate import build_record
from .constants import FIELDNAMES
from .extract import extract_packet


def _process_one(pdf_path: str) -> dict:
    path = Path(pdf_path)
    packet = extract_packet(path, case_id=path.stem)
    record = build_record(packet)
    # Drop internal keys
    return {k: record[k] for k in FIELDNAMES}


def iter_pdfs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.pdf"))


def run_pipeline(
    input_dir: Path,
    output_path: Path,
    workers: int = 4,
) -> int:
    pdfs = iter_pdfs(Path(input_dir))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    # Resume support: keep already-written JSONL rows if present.
    if output_path.exists():
        try:
            with output_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    cid = row.get("case_id")
                    if cid:
                        results[cid] = row
        except Exception:
            results = {}
    pending = [p for p in pdfs if p.stem not in results]
    if not pending and results:
        return len(results)

    def _flush() -> None:
        with output_path.open("w", encoding="utf-8") as f:
            for case_id in sorted(results):
                row = {
                    **results[case_id],
                    "confidence": float(results[case_id]["confidence"]),
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if workers <= 1 or len(pending) <= 1:
        for i, pdf in enumerate(pending, 1):
            rec = _process_one(str(pdf))
            results[rec["case_id"]] = rec
            if i % 25 == 0:
                _flush()
                print(f"checkpoint {len(results)}/{len(pdfs)}", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, str(pdf)): pdf for pdf in pending}
            done = 0
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                    results[rec["case_id"]] = rec
                except Exception as exc:  # noqa: BLE001
                    pdf = futures[fut]
                    print(f"WARN: failed {pdf.name}: {exc}", file=sys.stderr)
                done += 1
                if done % 25 == 0:
                    _flush()
                    print(f"checkpoint {len(results)}/{len(pdfs)}", file=sys.stderr)

    _flush()
    return len(results)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        raise SystemExit("usage: solution.py <input_pdf_dir> <output_predictions_path>")
    input_dir, output_path = argv
    # Use up to 4 workers to match scoring CPU allotment
    n = run_pipeline(Path(input_dir), Path(output_path), workers=4)
    print(f"wrote {n} predictions to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
