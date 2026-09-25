# Trakheesi Compliance Detector

FastAPI service that checks real-estate listing images + ad text against
Dubai's Trakheesi (Abu Dhabi: Madhmoun) advertising rules — permit-number
OCR match, unauthorized watermark detection (YOLOv8n, ONNX Runtime), and
duplicate/stock photo detection (perceptual hashing).

## Environment
- Python 3.13, venv at `venv/`. Activate: `source venv/bin/activate`
- `requirements.txt` = full dev/training set (includes torch, ultralytics)
- `requirements-serve.txt` = runtime-only subset used by the Docker image —
  keep these two in sync for overlapping runtime deps only

## Common commands
- Run tests: `pytest -v` (should be 63 passing)
- Run API locally: `uvicorn src.api:app --reload`
- Deploy: see `DEPLOY_AZURE.md`

## Notes
- OCR backend auto-picks Azure Vision when `AZURE_VISION_KEY` is set,
  otherwise falls back to local Tesseract (needs the `tesseract` binary
  on PATH) — see `src/ocr_permit.py`
- Prefer minimal, targeted fixes over rewrites
- Don't touch `venv/`, `__pycache__/`, `.pytest_cache/`, or the checked-in
  `yolov8n.pt` base weights unless asked
