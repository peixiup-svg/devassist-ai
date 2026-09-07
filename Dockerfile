FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVASSIST_ROOT=/app

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data/sample ./data/sample
COPY data/evaluation ./data/evaluation
COPY artifacts ./artifacts

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    addgroup --system devassist && \
    adduser --system --ingroup devassist devassist && \
    mkdir -p /app/data/runtime && \
    chown -R devassist:devassist /app

USER devassist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "devassist.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
