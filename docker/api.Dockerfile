# ---- builder ----
FROM python:3.13-slim@sha256:84a57da03fbb4a77e8769a3d5b692ee8f1d43a319eb6eee7c2e0b39caf406bb8 AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ---- runtime ----
FROM python:3.13-slim@sha256:84a57da03fbb4a77e8769a3d5b692ee8f1d43a319eb6eee7c2e0b39caf406bb8
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

LABEL org.opencontainers.image.title="medical-triage-api" \
      org.opencontainers.image.description="Bilingual RAG medical triage HTTP API" \
      org.opencontainers.image.source="https://github.com/etahir2005/medical-triage-assistant"

RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser api.py .

ENV PATH="/opt/venv/bin:$PATH" \
    HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_OFFLINE=1

RUN mkdir -p $HF_HOME && chown -R appuser:appuser $HF_HOME
USER appuser
RUN HF_HUB_OFFLINE=0 python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]