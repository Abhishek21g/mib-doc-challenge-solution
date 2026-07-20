#!/usr/bin/env bash
set -euo pipefail

input_dir="${1:?usage: run.sh <input_pdf_dir> <output_path>}"
output_path="${2:?usage: run.sh <input_pdf_dir> <output_path>}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TESSDATA_PREFIX="${TESSDATA_PREFIX:-/usr/share/tesseract-ocr/5/tessdata}"

python3 /app/solution.py "$input_dir" "$output_path"
