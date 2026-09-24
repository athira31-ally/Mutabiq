# 01 · Listing Image & Trakheesi Compliance Detector

**Status:** 🟢 **Live on Azure Container Apps** — [try the demo](https://trakheesi-api.victoriousriver-467d20dd.uaenorth.azurecontainerapps.io) (click a sample listing or upload your own; the first load after idle takes ~20 s while it scales up from zero).

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
| FastAPI service + web demo | `src/api.py`, `demo/` | `/` demo page, `/check-listing`, `/check-sample/{id}`, `/health` |
| Dockerfile + Azure Container Apps deploy script | `Dockerfile`, `scripts/azure_deploy.sh` | Serve-only image (no training deps) — see [`DEPLOY_AZURE.md`](DEPLOY_AZURE.md) |

**26 tests** across 7 modules (`tests/`) — run with `pytest -v`.

## Live demo

**https://trakheesi-api.victoriousriver-467d20dd.uaenorth.azurecontainerapps.io**

The demo page runs five real-photo sample listings through the full pipeline with one click — one per outcome:

| Sample | What's wrong | Verdict |
|---|---|---|
| Compliant listing | nothing — permit on the photo matches the ad | ✅ pass |
| Unauthorised watermark | another brokerage's wordmark on the photo | 🟡 review · `WATERMARK_DETECTED` |
| Permit doesn't match the ad | photo shows 1239982634, ad claims 7169578165 | 🟡 review · `PERMIT_MISMATCH` |
| No permit on the listing | no permit number anywhere | 🔴 fail · `PERMIT_MISSING` |
| Photo reused by another agent | sample 1's photo, re-cropped, posted by a different agent | 🟡 review · `DUPLICATE_PHOTO` |

Each verdict is pinned by `tests/test_demo_samples.py`. Deployment: GitHub Actions builds the serve-only image to `ghcr.io`, and `scripts/deploy_live.sh` runs it on Azure Container Apps (scale-to-zero, ~$0 when idle). You can also upload your own listing photos, or call `POST /check-listing` directly (docs at `/docs`).

### What going live taught me

Putting the model in front of new images surfaced three real issues, now fixed or documented:

1. **Train/serve skew from a font path.** The synthetic-watermark generator loads a Linux font (`DejaVuSans-Bold.ttf`). The training set was generated on macOS, where that path doesn't exist, so every training logo silently fell back to Pillow's small default font. The model therefore learned *small text wordmarks*: on fresh composites it detects **40/48** default-font marks but only **10/48** large bold ones. The demo uses training-style marks; **next step:** bundle a few open-licence fonts in the repo and retrain with varied fonts, sizes and styles.
2. **Busy photos hid the permit from OCR.** Tesseract's page-layout step missed a clearly printed permit banner on a cluttered living-room photo. Fix: if the full-image pass finds no permit, re-read the top and bottom bands (where permit badges sit) enlarged — `src/ocr_permit.py`.
3. **A threading bug in the duplicate index.** The SQLite connection was tied to the thread that created it, but FastAPI serves requests from several threads — the first request worked and later ones would crash. Fix: cross-thread connection + lock, with a concurrency test.

The detector's recall-first threshold (0.10) is unchanged by design: a false alarm only means "review". The only 5 clean images in the 40-image validation set include 2 false positives, so a larger negative set is also on the list.

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

Object detection (YOLOv8 fine-tuning), ONNX export and ONNX Runtime inference, synthetic training-data generation for a class with no public dataset, OCR (Azure AI Vision Read API + Tesseract, pluggable backends), fuzzy string matching for noisy-OCR cross-checks, perceptual hashing (pHash) for near-duplicate image detection, rule-based decision engines, FastAPI service design, pytest-driven development, Docker containerization, Azure Container Apps deployment (GitHub Actions image build to ghcr.io, scale-to-zero), a live web demo.
