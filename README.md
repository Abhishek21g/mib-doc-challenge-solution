# MIB Doc Challenge — Offline Intake Pipeline

Air-gapped PDF intake pipeline for [8090's MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

## What it does

1. Extracts trusted visible text from each PDF (ignores white/hidden prompt-injection layers).
2. OCRs scan-heavy pages via Tesseract when the text layer is empty or incomplete.
3. Resolves fields with document-type precedence (manual note → intake → biometric → sponsor → registry → fee).
4. Applies an adjudication policy inferred from the public field manual + labeled train examples.
5. Emits calibrated `APPROVED` / `DENIED` / `NEEDS_REVIEW` decisions as JSONL.

## Run offline (Docker)

```bash
docker build -t mib-submission .
mkdir -p /tmp/mib-output
docker run --rm --network none \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/tmp/mib-output,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

## Local (dev)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# system dependency: tesseract-ocr
python3 solution.py /path/to/pdfs /tmp/predictions.jsonl
```

## Design notes

- Hidden `SYSTEM:` / white-text answer keys are never trusted.
- `SAMPLE DENIAL` watermarks are ignored.
- Manual adjudicator findings override form fields when present.
- Revoked sponsors, embargo worlds, `TRANSIT-7`, unpaid fees, and disqualifying risk flags hard-deny.
- Review-only flags and weak/conflicting evidence route to `NEEDS_REVIEW`.
