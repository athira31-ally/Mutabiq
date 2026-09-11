import os

import pytest
from fastapi.testclient import TestClient

from src.dup_hash import DuplicatePhotoIndex
from src.ocr_permit import MockOCR
from src.pipeline import CompliancePipeline, ListingBundle

MODEL_PATH = "models/watermark_yolov8n.onnx"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason="trained model not present — run train_watermark_yolo.py first"
)


def test_pipeline_end_to_end_pass(tmp_image_with_text):
    img = tmp_image_with_text("Permit # 5551239987 valid listing")
    pipeline = CompliancePipeline(
        watermark_model_path=MODEL_PATH,
        dup_index=DuplicatePhotoIndex(":memory:"),
        ocr_backend=MockOCR("Permit # 5551239987 valid listing"),
    )
    bundle = ListingBundle(
        listing_id="L1", agent_id="A1", image_paths=[img], claimed_permit_number="5551239987"
    )
    report = pipeline.check_listing(bundle)
    assert report.status in ("pass", "review")  # watermark model may or may not fire on plain text image
    assert report.permit_check.matched_claimed is True


def test_pipeline_flags_missing_permit(tmp_image):
    pipeline = CompliancePipeline(
        watermark_model_path=MODEL_PATH,
        dup_index=DuplicatePhotoIndex(":memory:"),
        ocr_backend=MockOCR("no permit info at all"),
    )
    bundle = ListingBundle(listing_id="L2", agent_id="A1", image_paths=[tmp_image])
    report = pipeline.check_listing(bundle)
    assert report.status == "fail"


def test_api_health():
    from src.api import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}