FROM python:3.12.11-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

# Keep every common native/Python thread pool inside the four-vCPU scoring limit.
# Python bytecode and user caches are disabled because the container root is
# read-only at runtime. Any future scratch data belongs under /tmp.
ENV BLIS_NUM_THREADS=4 \
    HOME=/tmp \
    MALLOC_ARENA_MAX=4 \
    MIB_MAX_WORKERS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    OC_DISABLE_DOT_ACCESS_WARNING=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    TOKENIZERS_PARALLELISM=false \
    VECLIB_MAXIMUM_THREADS=4

WORKDIR /app

ARG TESSERACT_VERSION=5.3.0-2
ARG TESSERACT_DATA_VERSION=1:4.1.0-2
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      "tesseract-ocr=${TESSERACT_VERSION}" \
      "tesseract-ocr-eng=${TESSERACT_DATA_VERSION}" \
      "tesseract-ocr-osd=${TESSERACT_DATA_VERSION}" \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /app/requirements.lock
RUN python3 -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --no-deps \
      --require-hashes \
      --requirement /app/requirements.lock \
    && groupadd --gid "${APP_GID}" mib \
    && useradd \
      --uid "${APP_UID}" \
      --gid "${APP_GID}" \
      --home-dir /tmp \
      --no-create-home \
      --shell /usr/sbin/nologin \
      mib

COPY run.sh solution.py /app/
COPY mib_pipeline /app/mib_pipeline
COPY third_party_licenses /app/third_party_licenses
RUN chmod 0555 /app/run.sh /app/solution.py \
    && chmod -R a=rX /app/mib_pipeline \
    && chmod -R a=rX /app/third_party_licenses \
    && chmod 0444 /app/requirements.lock

USER mib:mib

ENTRYPOINT ["/app/run.sh"]
