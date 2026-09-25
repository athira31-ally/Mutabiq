# Architecture — Mutabiq, Trakheesi compliance checker for listing images

## 1. Problem statement

Before a Dubai property listing goes live, an agency or portal needs to know whether it will trip a Trakheesi (Dubai Land Department) or Madhmoun (Abu Dhabi) advertising violation. Confirmed rules from the published guidance:

- The permit number must appear on the advert, and must be current (not expired).
- The advert's stated details (price, property specs) must match the approved permit.
- Resale listings need a valid Form A (marketing agreement) on file; off-plan needs an NOC.
- Dubai (Trakheesi) and Abu Dhabi (Madhmoun) are separate, unconnected systems — a permit valid in one has no standing in the other.

Fines start at AED 50,000, with listing removal or licence suspension on repeat violations. The destination portal (Bayut, Property Finder, dubizzle) runs its own Trakheesi check before publishing, so this tool's job is to **catch violations before submission** — reducing rejected listings and the fine exposure of a mismatched or missing permit — plus catch two things portals don't check for: reused/stolen photos and unauthorized branding.

## 2. Scope for v1

In scope:
- Permit-number presence and legibility check (OCR)
- Permit-number format validation (regex, once the real format is confirmed — see Open Risks)
- Permit-number vs. ad-text cross-check (does the OCR'd number match what the ad claims?)
- Unauthorized broker watermark/logo detection on listing photos
- Duplicate/reused-photo detection across the ingested listing set

Out of scope for v1 (stretch/future):
- Live validation against DLD's actual Trakheesi database (no confirmed public lookup API — see Open Risks)
- Form A / NOC document verification (requires document-level parsing, not just image/text)
- Face/license-plate privacy blurring (useful add-on, not core to the compliance/fine story)

## 3. Pipeline

```
listing bundle (images[], ad_text, claimed_permit_number, property_details)
        │
        ▼
┌───────────────────┐   ┌────────────────────────┐   ┌───────────────────────┐
│ 1. Permit OCR &    │   │ 2. Watermark / logo    │   │ 3. Duplicate-photo    │
│    validation      │   │    detector             │   │    detector            │
│  (Azure AI Vision   │   │  (YOLO, ONNX runtime)   │   │  (perceptual hash      │
│   Read API + regex) │   │                         │   │   index lookup)        │
└─────────┬──────────┘   └───────────┬─────────────┘   └───────────┬───────────┘
          │                          │                             │
          └──────────────┬───────────┴─────────────┬───────────────┘
                          ▼                         ▼
                 ┌──────────────────────────────────────┐
                 │ 4. Rule engine / scorer                │
                 │  combines flags → structured report    │
                 └───────────────────┬────────────────────┘
                                     ▼
                         { status: pass | review | fail,
                           violations: [...],
                           confidence per check }
```

### 3.1 Permit OCR & validation

- Run Azure AI Vision's Read API over each listing image to extract all text regions (permit numbers are typically overlaid on the image itself, not just in the ad copy).
- Regex-match extracted text against the Trakheesi/Madhmoun permit format (format TBD — see Open Risks).
- Compare the extracted number against `claimed_permit_number` from the listing metadata; flag on mismatch.
- Hard fail if no permit-shaped string is found anywhere in the bundle.

### 3.2 Watermark / unauthorized-logo detector

- Fine-tune a YOLOv8n model (ONNX export — same inference path as SmartPoseEdge/Motivision) to detect logo/watermark bounding boxes on property photos.
- Training data doesn't exist publicly, so bootstrap synthetically: overlay a set of known brokerage logos (collected as PNGs) onto clean property photos at randomized position/scale/opacity to generate positive examples, mixed with clean negatives.
- v1 target: detect *presence of any watermark*, not classify *whose* watermark it is — classification-by-brokerage is a stretch goal once there's a real logo dataset.

### 3.3 Duplicate-photo detector

- Compute a perceptual hash (pHash or dHash via the `imagehash` library) for every ingested image.
- Maintain an index (SQLite for v1; FAISS if the corpus grows) keyed by hash.
- On a new image, query nearest neighbors by Hamming distance; flag matches under a threshold, especially when the match belongs to a *different* listing ID or agent — that's the signal for a reused or stolen photo, not just the same agent re-uploading their own shot.

### 3.4 Rule engine

Combines the checks into one report:

| Condition | Severity |
|---|---|
| No printed permit number and no Trakheesi permit QR code | Hard fail |
| Permit QR only (portal style), no printed number | OK — if the QR link carries a permit number, it must match the ad (else Review) |
| Permit QR belongs to a different listing than the pasted link | Review (`PERMIT_QR_OTHER_LISTING`) |
| Watermark that reads as the listing's registered agency | Allowed — reported as own branding, not a violation |
| Permit number present but malformed | Hard fail |
| Permit number present, valid format, but mismatches ad text | Review |
| Watermark detected | Review |
| Duplicate photo matched to a different listing/agent | Review |
| All checks clean | Pass |

## 4. Serving layer

FastAPI service wrapping the three checks + rule engine, with ONNX Runtime handling the YOLO model — the same deployment shape as SmartPoseEdge. Package as an Azure Function for a lightweight per-listing check, or Azure Container Apps if the FastAPI service needs to stay warm (larger model, batch throughput).

## 5. Data plan

- **Seed validation set:** ~100–200 real Bayut/Property Finder listing images with their publicly displayed permit numbers, collected manually for v1 rather than scraped at scale — see Open Risks on ToS.
- **Watermark training set:** synthetic overlays as described in 3.2.
- **Permit format reference:** none found in initial research (see below) — needs to come from the seed set itself.

## 6. Evaluation

- **OCR/permit extraction:** exact-match rate against the hand-labeled seed set.
- **Watermark detector:** precision/recall, weighted toward recall — a missed watermark (false negative) is the costlier failure for the actual fine-avoidance use case.
- **Duplicate detector:** precision at the chosen Hamming-distance threshold.
- **End-to-end:** % of seed-set listings correctly triaged into pass/review/fail.
