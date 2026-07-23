#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: run.sh <input_pdf_dir> <output_predictions_path>" >&2
  exit 64
fi

exec python3 -I -B /app/solution.py "$1" "$2"
