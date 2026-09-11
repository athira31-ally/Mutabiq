# 01 · Listing Image & Trakheesi Compliance Detector

**Status:** 🟡 MVP built — full pipeline implemented and tested. The Azure deployment path is built and documented but hasn't been pointed at a live subscription yet, so there is no public URL yet.

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
| Real base-photo fetcher (Pexels API) | `src/fetch_stock_photos.py` | Downloads free-licensed property/interior photos to use as training backgrounds — see "Training data" below |
| Synthetic watermark-overlay generator | `src/synthetic_watermark_data.py` | Composites a synthetic brokerage wordmark onto a base photo, in YOLO label format; base photo can be procedural or a real downloaded one |
| YOLOv8n watermark detector, training + ONNX export | `src/train_watermark_yolo.py` | Trained model + metrics committed under `models/watermark_detector/` — see Results below |
| ONNX Runtime inference (no torch dependency at serve time) | `src/watermark_detector.py` | Keeps the Docker image small — see `requirements-serve.txt` |
| Duplicate-photo detector (perceptual hashing) | `src/dup_hash.py` | SQLite-backed index (`data/dup_index.sqlite`) |
| Rule engine (combines checks into a compliance report) | `src/rule_engine.py` | Rule table in `ARCHITECTURE.md` §3.4 |
| Pipeline orchestration | `src/pipeline.py` | Wires the three checks + rule engine together |
| FastAPI service | `src/api.py` | `/health`, `/check-listing` |
| Dockerfile + Azure Container Apps deploy script | `Dockerfile`, `scripts/azure_deploy.sh` | Serve-only image (no training deps) — see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) |

**21 test functions** across 6 modules (`tests/`) — run with `pytest -v`.

## Live demo

The full deploy path — Dockerfile, `scripts/azure_deploy.sh`, and a step-by-step guide — is ready to run against an Azure subscription; see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) for the one-command version and the cost breakdown (roughly $5/month if left running, ~$0 if torn down between demos).

## Training data

No public dataset of "unauthorized broker watermark on a UAE listing photo" exists, so the training set is bootstrapped : a synthetic brokerage wordmark (never a real brand's actual logo, to sidestep trademark questions) is composited onto a base photo at randomized position/scale/opacity.

The base photos underneath are **real property/interior photographs**, fetched via `src/fetch_stock_photos.py` from the [Pexels API](https://www.pexels.com/api/) (240 photos in `data/raw/stock_photos/`) — free, no attribution required under the Pexels License, and used only as a training background, never republished as-is. `synthetic_watermark_data.py` also supports a procedural fallback (drawn gradient rooms, no external dependency) for offline iteration, which is what an earlier internal run used.

The committed model was trained on **250 training images / 40 validation images** built this way (`data/synthetic/watermark_yolo/`).

## Results

Final validation metrics after 60 epochs at 416px, on real-photo backgrounds (`models/watermark_detector/results.csv`):

| Metric | Value |
|---|---|
| Precision | 0.966 |
| Recall | 0.824 |
| mAP50 | 0.908 |
| mAP50-95 | 0.668 |

Training took ~35 minutes on CPU (no GPU in this build environment). An earlier run on purely procedural (non-photographic) backgrounds scored higher on paper (mAP50 0.995) — real backgrounds are harder (varied lighting, clutter, furniture), so the lower number here is the more meaningful one for real-world performance.

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

- The overlaid watermark is still a generated wordmark, not a real competitor's logo (deliberately, to avoid trademark issues) — it generalizes to "text/logo-shaped region on a photo," which is the right shape of signal but will need real broker logos to sharpen precision before it's production-ready.
- `PERMIT_REGEX` in `ocr_permit.py` is a provisional format inferred from public Trakheesi-checker tools, not an official DLD spec — see `ARCHITECTURE.md`, "Open risks."
- No live Trakheesi/Madhmoun permit-lookup API is confirmed to exist publicly; validation here is format + cross-check only, not live DLD verification.
- Not yet deployed to a live Azure endpoint — the deploy path is built and documented but unexercised against a real subscription.

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Fetch real base photos (optional but recommended — needs a free Pexels API key)
export PEXELS_API_KEY=your_key_here
python -m src.fetch_stock_photos --out data/raw/stock_photos --per-query 30

# 2. Generate the training set (250 train / 40 val, matching the committed model)
python -m src.synthetic_watermark_data --out data/synthetic/watermark_yolo \
    --n-train 250 --n-val 40 --real-photos-dir data/raw/stock_photos

# 3. Train + export the watermark detector (~35 min on CPU)
python -m src.train_watermark_yolo --data data/synthetic/watermark_yolo/dataset.yaml --epochs 60 --imgsz 416

# 4. Run tests
pytest -v

# 5. Serve
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
