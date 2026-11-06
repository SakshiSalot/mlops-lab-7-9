# House-price inference API (Part C).
#   docker build -t ml-inference-api:1.0 .
#   docker run --name ml-api -p 8000:8000 ml-inference-api:1.0
# Requires models/model.joblib + models/model_metadata.json in the build context
# (run `dvc pull` or `python -m src.train` first).

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=info

WORKDIR /app

# 1) Dependencies first: this layer is reused until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Application code and the model artifact (small, so it is baked into the image).
COPY src/ ./src/
COPY models/model.joblib models/model_metadata.json ./models/

# 3) Do not run as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Docker marks the container unhealthy if /health stops returning 200.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
