"""The listing agency's own logo is allowed; another broker's mark is not."""
import os

import pytest

from src.dup_hash import DuplicatePhotoIndex
from src.ocr_permit import MockOCR
from src.own_branding import brand_similarity, distinctive_tokens
from src.pipeline import CompliancePipeline, ListingBundle

MODEL = "models/watermark_yolov8n.onnx"
SAMPLE_B = "demo/samples/b_watermark.jpg"     # carries a "SKYLINE ESTATES" wordmark the detector finds


def test_distinctive_words_skip_what_every_agency_shares():
    assert distinctive_tokens("SEROVIA PROPERTIES L.L.C") == ["SEROVIA"]
    assert distinctive_tokens("Palm Realty Group") == ["PALM"]
    assert distinctive_tokens("Real Estate Properties LLC") == []


@pytest.mark.parametrize("ocr_text, own", [
    ("SEROVIA PROPERTIES", True),
    ("EROVIA\\nPROPERTIES", True),      # the stylised S is part of the logo's icon; OCR drops it
    ("pit FH ROVIA ~~ | 6 7", True),    # what Tesseract actually read from the real SEROVIA mark
    ("PROPERTIES", False),              # a generic word proves nothing
    ("DXB HOMES", False),
    ("", False),
])
def test_fuzzy_match_against_the_agency(ocr_text, own):
    score, _ = brand_similarity(ocr_text, "SEROVIA PROPERTIES L.L.C")
    assert (score >= 0.75) is own


@pytest.fixture
def pipeline_reading(tmp_path):
    if not os.path.exists(MODEL):
        pytest.skip("trained model not present")

    def make(ocr_text):
        return CompliancePipeline(MODEL, dup_index=DuplicatePhotoIndex(str(tmp_path / "idx.sqlite")),
                                  ocr_backend=MockOCR("Trakheesi Permit No. 6045128830\n" + ocr_text))
    return make


def test_own_logo_is_allowed(pipeline_reading):
    r = pipeline_reading("SKYLINE ESTATES").check_listing(ListingBundle(
        "L1", "A1", [SAMPLE_B], "6045128830", agency_name="SKYLINE ESTATES L.L.C")).to_dict()
    assert r["status"] == "pass" and r["watermarks_detected"] == 0
    assert r["own_branding"] and r["own_branding"][0]["matched"] == "SKYLINE"


def test_another_brokers_logo_is_flagged(pipeline_reading):
    r = pipeline_reading("SKYLINE ESTATES").check_listing(ListingBundle(
        "L2", "A2", [SAMPLE_B], "6045128830", agency_name="PALM REALTY GROUP")).to_dict()
    assert r["status"] == "review" and "WATERMARK_DETECTED" in [v["code"] for v in r["violations"]]
    assert r["own_branding"] == []


def test_without_a_known_agency_every_mark_counts(pipeline_reading):
    r = pipeline_reading("SKYLINE ESTATES").check_listing(ListingBundle("L3", "A3", [SAMPLE_B], "6045128830")).to_dict()
    assert r["watermarks_detected"] >= 1
