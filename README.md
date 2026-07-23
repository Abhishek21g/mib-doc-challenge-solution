# MIB Doc Challenge — Abhishek Enaguthi

Offline Docker submission for the [8090 MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge).

**Public-train score:** **130.72 / 150** (CFA 0) via official `evaluate.py`.

## Run

```bash
./run.sh /path/to/pdfs /path/to/predictions.jsonl
# or
MIB_MAX_WORKERS=2 python solution.py /path/to/pdfs /path/to/predictions.jsonl
```

## Stack

Render-first OCR (strobl MIT) + legal visible-text repairs + DIP-1 layout-consensus approval (no answer keys). See `MEMO.md` and `ATTRIBUTION.md`.
