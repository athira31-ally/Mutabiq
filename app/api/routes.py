from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import ComplianceReport, ListingMetadata
from app.services.pipeline import CompliancePipeline, decode_image

router = APIRouter()
_pipeline: CompliancePipeline | None = None


def get_pipeline() -> CompliancePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CompliancePipeline()
    return _pipeline


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "trakheesi-compliance-detector"}


@router.get("/v1/model")
def model_card() -> dict:
    from app.services.pipeline import MODEL_CARD

    return MODEL_CARD


@router.post("/v1/listings/inspect", response_model=ComplianceReport)
async def inspect_listing(
    listing: str = Form(..., description="JSON-encoded ListingMetadata"),
    images: list[UploadFile] = File(default_factory=list),
) -> ComplianceReport:
    try:
        meta = ListingMetadata.model_validate(json.loads(listing))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid listing metadata: {exc}") from exc

    decoded: list[tuple[str, object]] = []
    for upload in images:
        raw = await upload.read()
        frame = decode_image(raw)
        if frame is None:
            raise HTTPException(status_code=400, detail=f"Unreadable image: {upload.filename}")
        decoded.append((upload.filename or "upload.jpg", frame))

    return get_pipeline().inspect(meta, decoded)
