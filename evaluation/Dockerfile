FROM python:3.12.11-slim-bookworm

ARG APP_UID=10001
ARG APP_GID=10001

ENV HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp

WORKDIR /workspace

RUN groupadd --gid "${APP_GID}" evaluator \
    && useradd \
      --uid "${APP_UID}" \
      --gid "${APP_GID}" \
      --home-dir /tmp \
      --no-create-home \
      --shell /usr/sbin/nologin \
      evaluator

# Deliberately copy tooling and schemas only. Labels, PDFs, split manifests,
# reports, and predictions must be supplied through read-only/writable mounts.
COPY devtools /workspace/devtools
COPY mib_pipeline /workspace/mib_pipeline
COPY schemas /workspace/schemas
COPY scripts /workspace/scripts

RUN chmod -R a=rX /workspace

USER evaluator:evaluator

ENTRYPOINT ["python3", "/workspace/scripts/evaluation_harness.py"]
