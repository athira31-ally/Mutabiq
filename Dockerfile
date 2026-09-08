FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models
COPY data/fixtures ./data/fixtures
COPY scripts ./scripts

ENV PYTHONUNBUFFERED=1
ENV YOLO_ONNX_PATH=models/yolov8n_watermark.onnx
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
