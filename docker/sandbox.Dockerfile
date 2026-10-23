# Locked-down image for running untrusted, model-written code.
# Runtime hardening (no network, read-only root FS, cap-drop ALL, limits) is applied
# by harness.sandbox.docker at container start; this image only provides the runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp

RUN pip install --no-cache-dir "pandas==2.2.3" "numpy==2.1.3" \
 && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox \
 && mkdir -p /workspace /data \
 && chown sandbox:sandbox /workspace

USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["python", "-I", "-B"]
