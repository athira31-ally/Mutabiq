# Production image for the Trakheesi Compliance Detector API.
#
# Build (locally, optional — `az containerapp up --source .` in azure_deploy.sh
# builds this in the cloud via ACR Tasks, so you don't need Docker installed
# to deploy):
#   docker build -t trakheesi-api .
#   docker run -p 8000:8000 trakheesi-api
#
# Deliberately serve-only: no ultralytics/torch (training deps). The model is
# already trained and exported to models/watermark_yolov8n.onnx, committed to
# the repo, and just gets copied in below.

FROM python:3.11-slim

# tesseract-ocr: the binary pytesseract shells out to, for the local/offline
# OCR fallback (TesseractOCR backend). Kept in the image even though the
# Azure deployment will normally use AzureVisionOCR instead, so the image
# still works standalone (e.g. running it on a laptop with no Azure creds).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (separate layer) so `docker build` only re-installs
# packages when requirements-serve.txt actually changes, not on every code edit.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Application code + the trained model.
COPY src/ ./src
COPY models/ ./models

# Writable scratch dir for the duplicate-photo SQLite index. On Azure
# Container Apps' Consumption plan the container filesystem is ephemeral
# (wiped on restart / scale-to-zero) — fine for a portfolio demo. For real
# persistence across restarts, mount an Azure Files share at /app/data
# instead (see DEPLOY_AZURE.md, "Optional: persistent storage").
RUN mkdir -p /app/data

ENV WATERMARK_MODEL_PATH=/app/models/watermark_yolov8n.onnx
ENV DUP_INDEX_PATH=/app/data/dup_index.sqlite
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
