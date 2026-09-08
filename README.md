# Trakheesi Compliance Detector

UAE real-estate listing compliance (2026). A FastAPI microservice that checks Dubai Land Department **Trakheesi** advertising permits on listing creatives, detects agency **watermarks/logos** with a fine-tuned YOLOv8n model served through **ONNX Runtime**, and flags **stolen or reused photos** with perceptual hashes.

Designed for Azure Container Apps. Training photos are real apartment interiors from [Pexels](https://www.pexels.com/).

## What it checks

| Check | What happens |
| --- | --- |
| Permit format / expiry | Listing metadata must carry a plausible DLD/Trakheesi reference that has not expired. |
| OCR vs metadata | Tesseract (or Azure AI Vision when configured) reads on-image text. A field-aware fuzzy matcher reconciles that text with the declared permit, broker, and developer. |
| Watermark / logo | YOLOv8n ONNX detector flags competitor or agency marks overlaid on photos. |
| Duplicate photos | pHash Hamming distance against an in-process index (stolen listing photography). |

RERA/DLD rules this maps to: every public advertisement must show the Trakheesi number issued for that property and advertiser; permit numbers must not be reused across properties; misleading branding and duplicate/unauthorised photography are advertising-conduct risks. This service is a **pre-publish gate**, not a substitute for DLD Verify License and Permits.

## Fuzzy matcher (redesign)

The first design required the **entire OCR blob** to equal `listing.permit_number`. A real Dubai Marina portal creative broke that:

`AED 2,150,000  +971 50 123 4567  Permit  7113984521  Horizon Gate Real Estate`

Price and phone dominate the string, so exact match always failed, and a naive "first long digit run" often selected the mobile number.

The engine now:

1. Strips AED amounts and `+971` mobiles from the candidate pool.
2. Prefers tokens next to `Permit` / `Trakheesi` / Arabic permit labels.
3. Folds OCR confusions (`O`/`0`, `I`/`1`).
4. Scores permit identity separately from broker/developer `token_set_ratio`.

Fixture: `data/fixtures/marina_listing.json`.

## Detector

Held-out validation after fine-tuning YOLOv8n on Pexels interiors with **synthetic generic marks** (no competitor trademarks):

| Metric | Value |
| --- | --- |
| mAP50 | 0.908 |
| Precision | 0.966 |
| Recall | 0.824 |
| Accuracy at conf 0.25 | 0.80 |
| Accuracy at swept conf **0.42** | **0.85** |

Serve-time inference does **not** import Ultralytics. `OnnxDetector` reads `N x C x H x W` from the model file and letterboxes to that size.

Replace `models/yolov8n_watermark.onnx` with your exported weights. A dummy graph is enough for CI and cold starts:

```bash
python scripts/create_dummy_onnx.py
```

## API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/create_dummy_onnx.py
uvicorn app.main:app --reload --port 8080
```

- `GET /health`
- `GET /v1/model`
- `POST /v1/listings/inspect` -- multipart `listing` (JSON `ListingMetadata`) + `images`

```bash
curl -s localhost:8080/v1/listings/inspect \
  -F 'listing={"listing_id":"dxb-1","permit_number":"7113984521","broker_name":"Horizon Gate Real Estate"};type=application/json' \
  -F images=@/path/to/interior.jpg
```

## Training data (Pexels to synthetic labels)

There is no public dataset of brokerage watermarks that is both legal to redistribute and representative of UAE interiors. The bootstrap:

```bash
export PEXELS_API_KEY=...
python -m training.fetch_pexels --api-key "$PEXELS_API_KEY"
python -m training.build_dataset
pip install ultralytics
python -m training.train --epochs 40
```

Overlays are geometric marks plus generic tokens (`MARK`, `AGCY`, ...), not real agency wordmarks.

## Tests

```bash
pytest
```

26 automated tests cover permit rules, OCR candidate extraction, the redesigned fuzzy engine, pHash duplicates, ONNX input-shape loading, the confidence sweep, synthetic labels, the pipeline, and the HTTP API.

## Docker / Azure Container Apps

```bash
docker compose up --build
```

Bicep: `infra/containerapp.bicep`. Set `AZURE_VISION_ENDPOINT` and `AZURE_VISION_KEY` to prefer Azure AI Vision Read OCR; otherwise the container uses Tesseract.

## Configuration

See `.env.example`. `DETECT_CONF_THRESHOLD` defaults to `0.42` from the precision/recall sweep.
