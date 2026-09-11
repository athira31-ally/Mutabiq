# 01 · Listing Image & Trakheesi Compliance Detector

**Status:** 🟡 MVP built — full pipeline implemented and tested against synthetic data. The Azure deployment path is built and documented but hasn't been pointed at a live subscription yet, so there is no public URL yet.

Before a Dubai property listing goes live, an agency or portal needs to know whether it will trip a **Trakheesi** (Dubai Land Department) or **Madhmoun** (Abu Dhabi) advertising violation — a missing or illegible permit number, a permit number that doesn't match the ad, an unauthorized broker watermark on the photos, or a duplicate/stock photo reused across listings. DLD fines for Trakheesi violations start at **AED 50,000**, with listing removal or licence suspension on repeat offences, and this maps to a real, funded product category — several UAE proptech vendors already sell "Trakheesi validation" as a paid add-on.

This project catches those four violation types **before submission**, as a single API call.

Full design rationale, the rule table, and open risks are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## How it works

```
listing bundle (images[], ad_text, claimed_permit_number, listing/agent id)
        │
        ▼
┌────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ 1. Permit OCR &     │   │ 2. Watermark / logo    │   │ 3. Duplicate-photo     │
│    cross-check      │   │    detector             │   │    detector            │
│ (Azure AI Vision /  │   │  (YOLOv8n, ONNX Runtime)│   │  (perceptual hashing)  │
│  Tesseract, fuzzy   │   │                         │   │                        │
│  match)             │   │                         │   │                        │
└─────────┬───────────┘   └───────────┬─────────────┘   └───────────┬────────────┘
          │                           │                             │
          └─────────────┬─────────────┴─────────────┬───────────────┘
                         ▼                           ▼
                ┌────────────────────────────────────────┐
                │ 4. Rule engine                          │
                │  combines flags into one report          │
                └───────────────────┬────────────────────┘
                                    ▼
                        { status: pass | review | fail,
                          violations: [...] }
```

## What's implemented

| Component | File | Notes |
|---|---|---|
| Permit OCR + validation (Azure Vision / Tesseract / Mock backends) | `src/ocr_permit.py` | Backend auto-picked at runtime: Azure Vision when `AZURE_VISION_KEY` is set, else local Tesseract |
| Synthetic watermark training-data generator | `src/synthetic_watermark_data.py` | Procedurally overlays brokerage-style logos onto generated property photos — no public dataset exists for this, see `ARCHITECTURE.md` |
| YOLOv8n watermark detector, training + ONNX export | `src/train_watermark_yolo.py` | Trained model + metrics committed under `models/watermark_detector/` — see Results below |
| ONNX Runtime inference (no torch dependency at serve time) | `src/watermark_detector.py` | Keeps the Docker image small — see `requirements-serve.txt` |
| Duplicate-photo detector (perceptual hashing) | `src/dup_hash.py` | SQLite-backed index (`data/dup_index.sqlite`) |
| Rule engine (combines checks into a compliance report) | `src/rule_engine.py` | Rule table in `ARCHITECTURE.md` §3.4 |
| Pipeline orchestration | `src/pipeline.py` | Wires the three checks + rule engine together |
| FastAPI service | `src/api.py` | `/health`, `/check-listing` |
| Dockerfile + Azure Container Apps deploy script | `Dockerfile`, `scripts/azure_deploy.sh` | Serve-only image (no training deps) — see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) |

**21 test functions** across 6 modules (`tests/`) — run with `pytest -v`.

## Live demo

Not deployed yet. The full deploy path — Dockerfile, `scripts/azure_deploy.sh`, and a step-by-step guide — is ready to run against an Azure subscription; see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) for the one-command version and the cost breakdown (roughly $5/month if left running, ~$0 if torn down between demos).

## Results

The watermark detector was fine-tuned (YOLOv8n → ONNX) on synthetic data — procedurally generated property photos with randomized brokerage-logo overlays, 250 training images and 40 held-out validation images (`src/synthetic_watermark_data.py`, since no public watermark-violation dataset exists).

Final validation metrics after 60 epochs at 416px (`models/watermark_detector/results.csv`):

| Metric | Value |
|---|---|
| Precision | 0.966 |
| Recall | 0.824 |
| mAP50 | 0.908 |
| mAP50-95 | 0.668 |

Training took ~35 minutes on CPU (no GPU in this build environment).

### Illustrative example

A permit mismatch, exercised directly by `tests/test_ocr_permit.py::test_check_permit_review_on_mismatch` — an ad claims permit `7169578165`, but OCR extracts `1239982634` from the listing image:

```json
{
  "listing_id": "demo-1",
  "status": "review",
  "violations": [
    {
      "code": "PERMIT_MISMATCH",
      "severity": "review",
      "message": "OCR'd permit '1239982634' doesn't match claimed '7169578165' (similarity 0.10)."
    }
  ]
}
```

The rule engine treats a clean mismatch as **review** rather than an automatic hard fail, since OCR noise can produce false mismatches — see `ARCHITECTURE.md` §3.4 for the full pass/review/fail table.

## Known limitations

- The watermark detector is trained entirely on synthetic data. It generalizes to "text/logo-shaped region on a photo," which is the right shape of signal but will need real listing photos and real competitor logos to sharpen precision before it's production-ready.
- `PERMIT_REGEX` in `ocr_permit.py` is a provisional format inferred from public Trakheesi-checker tools, not an official DLD spec — see `ARCHITECTURE.md`, "Open risks."
- No live Trakheesi/Madhmoun permit-lookup API is confirmed to exist publicly; validation here is format + cross-check only, not live DLD verification.
- Not yet deployed to a live Azure endpoint — the deploy path is built and documented but unexercised against a real subscription.

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic training data (250 train / 40 val, matching the committed model)
python -m src.synthetic_watermark_data --out data/synthetic/watermark_yolo --n-train 250 --n-val 40

# 2. Train + export the watermark detector (~35 min on CPU)
python -m src.train_watermark_yolo --data data/synthetic/watermark_yolo/dataset.yaml --epochs 60 --imgsz 416

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

For a real public HTTPS deployment on Azure Container Apps with Azure AI Vision as the OCR backend, see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) — or run `./scripts/azure_deploy.sh` for the one-command version. `./scripts/azure_teardown.sh` tears it back down.

## Techniques used

Object detection (YOLOv8 fine-tuning), ONNX export and ONNX Runtime inference, synthetic training-data generation for a class with no public dataset, OCR (Azure AI Vision Read API + Tesseract, pluggable backends), fuzzy string matching for noisy-OCR cross-checks, perceptual hashing (pHash) for near-duplicate image detection, rule-based decision engines, FastAPI service design, pytest-driven development, Docker containerization, Azure Container Apps deployment (ACR Tasks cloud build, scale-to-zero, secrets management).
