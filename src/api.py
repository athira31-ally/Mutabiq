"""FastAPI service wrapping the compliance pipeline.

    uvicorn src.api:app --reload

Deploy target: Azure Functions (lightweight, per-listing) or Azure Container
Apps (if the service needs to stay warm for batch throughput) — see
ARCHITECTURE.md §4.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dup_hash import DuplicatePhotoIndex
from .pipeline import CompliancePipeline, ListingBundle

MODEL_PATH = os.environ.get("WATERMARK_MODEL_PATH", "models/watermark_yolov8n.onnx")
DUP_INDEX_PATH = os.environ.get("DUP_INDEX_PATH", "data/dup_index.sqlite")
ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo"                      # web demo page + sample listings (scripts/make_demo_samples.py)
SAMPLES_DIR = DEMO_DIR / "samples"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024            # per image, keeps the public demo from being abused

app = FastAPI(
    title="Trakheesi Compliance Detector",
    description="Flags Trakheesi/Madhmoun advertising violations in a listing before it's published.",
    version="0.1.0",
)

# Lets a plain HTML page (tools/manual_test.html, opened directly from disk
# as a file:// page — no dev server) call this API from the browser. Wide
# open (allow_origins=["*"]) is fine for a local/demo compliance-check API
# like this one; a consumer-facing production service taking real user data
# would want to scope this down to specific known origins instead.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: CompliancePipeline | None = None


def get_pipeline() -> CompliancePipeline:
    global _pipeline
    if _pipeline is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Watermark model not found at {MODEL_PATH}. Train it first: "
                "python -m src.train_watermark_yolo",
            )
        _pipeline = CompliancePipeline(
            watermark_model_path=MODEL_PATH,
            dup_index=DuplicatePhotoIndex(DUP_INDEX_PATH),
        )
        _seed_demo_index(_pipeline.dup_index)
    return _pipeline


def _samples() -> list[dict]:
    path = SAMPLES_DIR / "samples.json"
    return json.loads(path.read_text()) if path.exists() else []


def _seed_demo_index(index: DuplicatePhotoIndex) -> None:
    """The container filesystem is ephemeral (scale-to-zero wipes it), so after every cold start put the
    original of the 'reused photo' demo back in the index - sample E is then always caught as a re-post."""
    original = next((s for s in _samples() if s["id"] == "a_compliant"), None)
    if original and (SAMPLES_DIR / original["image"]).exists():
        index.add(image_id=f"{original['listing_id']}_0", image_path=str(SAMPLES_DIR / original["image"]),
                  listing_id=original["listing_id"], agent_id=original["agent_id"])


class ComplianceResponse(BaseModel):
    listing_id: str
    status: str
    violations: list[dict]
    permit: dict
    watermarks_detected: int
    duplicate_matches: int


@app.get("/health")
def health():
    return {"status": "ok"}


if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/", include_in_schema=False)
@app.get("/test", include_in_schema=False)
def demo_page():
    """The web demo: run the sample listings with one click, or upload your own."""
    page = DEMO_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="demo/index.html not found")
    return FileResponse(page)


@app.post("/check-sample/{sample_id}")
def check_sample(sample_id: str):
    """Run the full pipeline on one of the built-in demo listings (see demo/samples/samples.json)."""
    sample = next((s for s in _samples() if s["id"] == sample_id), None)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"Unknown sample {sample_id!r}")
    t0 = time.perf_counter()
    report = get_pipeline().check_listing(ListingBundle(
        listing_id=sample["listing_id"], agent_id=sample["agent_id"],
        image_paths=[str(SAMPLES_DIR / sample["image"])], claimed_permit_number=sample["claimed"]))
    return {**report.to_dict(), "expected": sample["expect"], "latency_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/check-listing", response_model=ComplianceResponse)
async def check_listing(
    listing_id: str | None = Form(None),
    agent_id: str = Form("web-visitor"),
    claimed_permit_number: str | None = Form(None),
    ad_text: str = Form(""),
    images: list[UploadFile] = File(...),
):
    pipeline = get_pipeline()
    listing_id = listing_id or f"WEB-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, upload in enumerate(images):
            data = await upload.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Each image must be under 8 MB.")
            path = Path(tmpdir) / f"img_{i}{Path(upload.filename or 'x.jpg').suffix[:5]}"
            path.write_bytes(data)
            image_paths.append(str(path))

        bundle = ListingBundle(
            listing_id=listing_id,
            agent_id=agent_id,
            image_paths=image_paths,
            claimed_permit_number=claimed_permit_number,
            ad_text=ad_text,
        )
        report = pipeline.check_listing(bundle)

    return report.to_dict()