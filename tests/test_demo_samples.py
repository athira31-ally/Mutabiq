"""The live demo: every built-in sample listing must produce exactly its intended verdict, and the web
endpoints (page, samples, one-click check, upload) must work end to end."""
import json
import os
import shutil

import pytest
from fastapi.testclient import TestClient

from src.dup_hash import DuplicatePhotoIndex
from src.pipeline import CompliancePipeline, ListingBundle

MODEL = "models/watermark_yolov8n.onnx"
SAMPLES = "demo/samples"
pytestmark = [
    pytest.mark.skipif(not os.path.exists(MODEL), reason="trained model not present"),
    pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary not installed"),
]
EXPECTED_CODES = {"a_compliant": [], "b_watermark": ["WATERMARK_DETECTED"], "c_mismatch": ["PERMIT_MISMATCH"],
                  "d_no_permit": ["PERMIT_MISSING"], "e_reused": ["DUPLICATE_PHOTO"]}


def _samples():
    return json.load(open(f"{SAMPLES}/samples.json"))


def test_each_sample_gets_its_intended_verdict():
    pipe = CompliancePipeline(MODEL, dup_index=DuplicatePhotoIndex(":memory:"))
    for s in _samples():          # in order: A registers the photo that E later re-posts
        r = pipe.check_listing(ListingBundle(s["listing_id"], s["agent_id"], [f"{SAMPLES}/{s['image']}"],
                                             s["claimed"])).to_dict()
        assert r["status"] == s["expect"], (s["id"], r)
        assert [v["code"] for v in r["violations"]] == EXPECTED_CODES[s["id"]], (s["id"], r)


@pytest.fixture
def client():
    import src.api as api
    api._pipeline = CompliancePipeline(MODEL, dup_index=DuplicatePhotoIndex(":memory:"))
    api._seed_demo_index(api._pipeline.dup_index)
    yield TestClient(api.app)
    api._pipeline = None


def test_demo_page_and_samples_are_served(client):
    assert "Trakheesi Compliance Detector" in client.get("/").text
    assert len(client.get("/samples/samples.json").json()) == 5
    assert client.get("/samples/a_compliant.jpg").status_code == 200


def test_reused_photo_is_caught_even_on_a_cold_start(client):
    # E runs first, before anyone ran A: the seeded index must still catch the re-post
    r = client.post("/check-sample/e_reused").json()
    assert r["status"] == "review" and r["duplicate_matches"] >= 1 and r["expected"] == "review"
    assert client.post("/check-sample/nope").status_code == 404


def test_upload_your_own_listing(client):
    with open(f"{SAMPLES}/c_mismatch.jpg", "rb") as f:
        r = client.post("/check-listing", data={"claimed_permit_number": "7169578165"},
                        files=[("images", ("photo.jpg", f, "image/jpeg"))]).json()
    assert r["status"] == "review" and r["listing_id"].startswith("WEB-")
    assert "1239982634" in r["permit"]["found_numbers"]
