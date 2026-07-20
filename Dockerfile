FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-eng \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY mib_pipeline /app/mib_pipeline
COPY solution.py /app/solution.py
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Read-only rootfs friendly: no writes under /app at runtime.
ENTRYPOINT ["/app/run.sh"]
