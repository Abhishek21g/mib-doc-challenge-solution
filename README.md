# MIB Doc Challenge Solution — Abhishek Enaguthi

Offline intake pipeline for the [8090 MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

**Public-train score:** **130.38 / 150** (CFA 0) via official `evaluate.py`.

## Run

```bash
docker build -t mib-submission .
docker run --rm --network none \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/path/to/out,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

Or locally: `python solution.py <input_pdf_dir> <output_predictions_path>`

## Policy

- No case-ID hardcodes / ground-truth tables
- No `SYSTEM:` / answer-key decoy features
- Offline-only (no network / LLM / VLM at score time)

See `MEMO.md` and `ATTRIBUTION.md`.
