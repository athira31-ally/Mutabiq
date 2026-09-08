import numpy as np

from app.models.schemas import BoundingBox, ListingMetadata
from app.services.duplicates import DuplicateIndex
from app.services.pipeline import CompliancePipeline


def _pipeline(ocr: str, boxes: list[BoundingBox] | None = None) -> CompliancePipeline:
    return CompliancePipeline(
        ocr_fn=lambda img: ocr,
        detect_fn=lambda img: boxes or [],
        duplicate_index=DuplicateIndex(path=None, threshold=8),
    )


def test_pipeline_compliant_pass(listing: ListingMetadata, interior_bgr: np.ndarray) -> None:
    report = _pipeline(
        "Permit 7113984521 Horizon Gate Real Estate Emaar Properties"
    ).inspect(listing, [("living.jpg", interior_bgr)])
    assert report.compliant is True
    names = {c.name for c in report.checks}
    assert names >= {"permit_format", "ocr_metadata_reconcile", "watermark_logo", "duplicate_photo"}


def test_pipeline_watermark_fail(listing: ListingMetadata, interior_bgr: np.ndarray) -> None:
    box = BoundingBox(x1=10, y1=10, x2=80, y2=40, confidence=0.93)
    report = _pipeline("Permit 7113984521 Horizon Gate Real Estate", [box]).inspect(
        listing, [("living.jpg", interior_bgr)]
    )
    assert report.compliant is False
    wm = next(c for c in report.checks if c.name == "watermark_logo")
    assert wm.passed is False


def test_pipeline_duplicate_fail(listing: ListingMetadata, interior_bgr: np.ndarray) -> None:
    index = DuplicateIndex(path=None, threshold=8)
    pipe_a = CompliancePipeline(
        ocr_fn=lambda img: "Permit 7113984521 Horizon Gate Real Estate",
        detect_fn=lambda img: [],
        duplicate_index=index,
    )
    listing_a = listing.model_copy(update={"listing_id": "listing-a"})
    pipe_a.inspect(listing_a, [("a.jpg", interior_bgr)])
    listing_b = listing.model_copy(update={"listing_id": "listing-b", "permit_number": "7113984521"})
    report = pipe_a.inspect(listing_b, [("b.jpg", interior_bgr)])
    dup = next(c for c in report.checks if c.name == "duplicate_photo")
    assert dup.passed is False
    assert report.compliant is False


def test_pipeline_missing_permit_fail(listing: ListingMetadata, interior_bgr: np.ndarray) -> None:
    report = _pipeline("AED 2,150,000 gym pool").inspect(listing, [("living.jpg", interior_bgr)])
    ocr = next(c for c in report.checks if c.name == "ocr_metadata_reconcile")
    assert ocr.passed is False
    assert report.compliant is False
