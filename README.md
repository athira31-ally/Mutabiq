# 01 · Listing image & Trakheesi compliance detector

**Status:** 🟡 MVP built — core pipeline implemented and tested, running on synthetic data

A pipeline that takes a listing image bundle plus its ad text and flags real, fine-worthy violations of Dubai's Trakheesi (or Abu Dhabi's Madhmoun) advertising rules: missing/illegible permit number, permit-number mismatch, unauthorized broker watermarks, and duplicate/stock photos reused across listings.

DLD fines for Trakheesi violations start at AED 50,000, with listing removal or licence suspension on repeat offences — this maps to a real, funded product category (several UAE proptech vendors already sell "Trakheesi validation" as a paid add-on).

Full design rationale, the rule table, and open risks are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## What's implemented

| Component | File | Status |
|---|---|---|
| Permit OCR + validation (Azure Vision / Tesseract / Mock backends) | `src/ocr_permit.py` | ✅ Implemented, tested with real Tesseract OCR |
| Synthetic watermark training data generator | `src/synthetic_watermark_data.py` | ✅ Implemented, generates YOLO-format dataset |
| YOLOv8n watermark detector, training + ONNX export | `src/train_watermark_yolo.py` | ✅ Trained — see results below |
| ONNX Runtime inference (no torch dependency at serve time) | `src/watermark_detector.py` | ✅ Implemented |
| Duplicate-photo detector (perceptual hashing) | `src/dup_hash.py` | ✅ Implemented |
| Rule engine (combines checks into a compliance report) | `src/rule_engine.py` | ✅ Implemented |
| Pipeline orchestration | `src/pipeline.py` | ✅ Implemented |
| FastAPI service | `src/api.py` | ✅ Implemented, live end-to-end tested |
| Azure Vision production OCR backend | `src/ocr_permit.py::AzureVisionOCR` | 🟡 Written and deployable — see below |
| Dockerfile + Azure Container Apps deploy | `Dockerfile`, `scripts/azure_deploy.sh` | ✅ Implemented — see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) |
| CI (tests on every push) | `.github/workflows/ci-compliance-detector.yml` | ✅ Implemented |

**19/19 tests passing** (`pytest -v`), including against the exact runtime-only dependency set (`requirements-serve.txt`) the Docker image ships.

## Live demo

Not deployed yet — the deploy path is fully built and ready to run (`DEPLOY_AZURE.md`), it just hasn't been pointed at a live Azure subscription yet. Once it is, the URL goes here.

## Results

Trained YOLOv8n for 25 epochs on 100 synthetic training images (procedurally generated property photos with randomized brokerage-logo overlays — see `src/synthetic_watermark_data.py` for why no real dataset exists yet), exported to ONNX, and evaluated on a 24-image held-out validation set:

- **mAP50: 0.995, Precision: 0.996, Recall: 1.0** (ultralytics validation)
- **24/24 correct pass/fail classification** on the val set through the custom ONNX Runtime inference wrapper (`watermark_detector.py`) — confirms the training → export → serve path is wired correctly end to end, not just correct inside ultralytics' own eval loop
- Training took ~5 minutes on CPU (no GPU in this build environment)

Real Tesseract OCR test (not synthetic — an actual rendered image, actually OCR'd): the engine misread "DLD-73920" as "DLO-73920" (D→O). The fuzzy cross-check (`difflib.SequenceMatcher`, 0.9 similarity threshold) correctly caught this as a **review**-severity mismatch rather than silently passing or hard-failing — validates the tolerance-band design against real OCR noise, not just clean synthetic text.

Full pipeline, live end to end through the FastAPI `/check-listing` endpoint with a real image upload:

```json
{
  "listing_id": "demo-1",
  "status": "review",
  "violations": [
    {"code": "PERMIT_MISMATCH", "severity": "review", "message": "OCR'd permit 'DLO-73920' doesn't match claimed 'DLD-73920' (similarity 0.88)."},
    {"code": "WATERMARK_DETECTED", "severity": "review", "message": "1 unauthorized watermark/logo region(s) detected."}
  ]
}
```

## Known limitations

- The watermark detector is trained entirely on synthetic data (procedural rooms + generated wordmark logos). It generalizes to "text/logo-shaped region in a photo," which is the right shape of signal but will need real listing photos + real competitor logos to sharpen precision before it's production-ready — it currently also fires on incidental image text, as seen in the demo above.
- `PERMIT_REGEX` in `ocr_permit.py` is a provisional format inferred from public Trakheesi-checker tools, not an official DLD spec — see `ARCHITECTURE.md`, "Open risks."
- No live Trakheesi/Madhmoun permit-lookup API is confirmed to exist publicly; validation here is format + cross-check only, not live DLD verification.

## Running it locally

```bash
pip install -r requirements.txt

# 1. Generate synthetic training data
python -m src.synthetic_watermark_data --out data/synthetic/watermark_yolo --n-train 100 --n-val 24

# 2. Train + export the watermark detector (~5 min on CPU)
python -m src.train_watermark_yolo --data data/synthetic/watermark_yolo/dataset.yaml --epochs 25 --imgsz 256

# 3. Run tests
pytest -v

# 4. Serve
uvicorn src.api:app --reload
```

## Running it in Docker / deploying to Azure

```bash
docker build -t trakheesi-api .
docker run -p 8000:8000 trakheesi-api
```

For a real public HTTPS deployment on Azure Container Apps with Azure AI Vision as the OCR backend, see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) — it explains every step, or run `./scripts/azure_deploy.sh` for the one-command version.

## Techniques used

Object detection (YOLOv8 fine-tuning), ONNX export and ONNX Runtime inference, synthetic training-data generation for a class with no public dataset, OCR (Azure AI Vision Read API + Tesseract, pluggable backends), fuzzy string matching for noisy-OCR cross-checks, perceptual hashing (pHash) for near-duplicate image detection, rule-based decision engines, FastAPI service design, pytest-driven development, Docker containerization, Azure Container Apps deployment (ACR Tasks cloud build, scale-to-zero, secrets management), CI via GitHub Actions.
