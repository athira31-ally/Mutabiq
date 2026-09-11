"""FastAPI service wrapping the compliance pipeline.

    uvicorn src.api:app --reload

Deploy target: Azure Functions (lightweight, per-listing) or Azure Container
Apps (if the service needs to stay warm for batch throughput) — see
ARCHITECTURE.md §4.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .dup_hash import DuplicatePhotoIndex
from .pipeline import CompliancePipeline, ListingBundle

MODEL_PATH = os.environ.get("WATERMARK_MODEL_PATH", "models/watermark_yolov8n.onnx")
DUP_INDEX_PATH = os.environ.get("DUP_INDEX_PATH", "data/dup_index.sqlite")
MANUAL_TEST_PAGE = Path(__file__).resolve().parent.parent / "tools" / "manual_test.html"

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
    return _pipeline


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


@app.get("/test", include_in_schema=False)
def manual_test_page():
    """Serves tools/manual_test.html directly from this API — so testing it
    is just http://127.0.0.1:8000/test (or the Azure URL + /test once
    deployed), one address, no separate file:// tab and no CORS to think
    about, since page and API are then the same origin.
    """
    if not MANUAL_TEST_PAGE.exists():
        raise HTTPException(status_code=404, detail="tools/manual_test.html not found")
    return FileResponse(MANUAL_TEST_PAGE)


@app.post("/check-listing", response_model=ComplianceResponse)
async def check_listing(
    listing_id: str = Form(...),
    agent_id: str = Form(...),
    claimed_permit_number: str | None = Form(None),
    ad_text: str = Form(""),
    images: list[UploadFile] = File(...),
):
    pipeline = get_pipeline()

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, upload in enumerate(images):
            path = Path(tmpdir) / f"img_{i}_{upload.filename}"
            path.write_bytes(await upload.read())
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