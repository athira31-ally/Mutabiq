"""End-to-end orchestration: listing bundle in, ComplianceReport out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .dup_hash import DuplicatePhotoIndex
from .ocr_permit import OCRBackend, check_permit, get_backend
from .rule_engine import ComplianceReport, evaluate
from .watermark_detector import WatermarkDetector


@dataclass
class ListingBundle:
    listing_id: str
    agent_id: str
    image_paths: list[str]
    claimed_permit_number: str | None = None
    ad_text: str = ""


class CompliancePipeline:
    def __init__(
        self,
        watermark_model_path: str | Path,
        dup_index: DuplicatePhotoIndex | None = None,
        ocr_backend: OCRBackend | None = None,
        watermark_imgsz: int | None = None,
        watermark_conf_threshold: float = 0.10,
    ):
        # watermark_imgsz=None (the default) means "read it from the model
        # file itself" — see WatermarkDetector.__init__. Pass an explicit
        # value here only to deliberately override that.
        self.watermark_detector = WatermarkDetector(
            watermark_model_path, imgsz=watermark_imgsz, conf_threshold=watermark_conf_threshold
        )
        self.dup_index = dup_index or DuplicatePhotoIndex()
        self.ocr_backend = ocr_backend or get_backend("auto")

    def check_listing(self, bundle: ListingBundle, register_images: bool = True) -> ComplianceReport:
        if not bundle.image_paths:
            raise ValueError("Listing bundle has no images.")

        primary_image = bundle.image_paths[0]
        permit_result = check_permit(
            primary_image, claimed_permit_number=bundle.claimed_permit_number, backend=self.ocr_backend
        )

        all_watermark_detections = []
        all_dup_matches = []
        for i, image_path in enumerate(bundle.image_paths):
            all_watermark_detections.extend(self.watermark_detector.detect(image_path))
            all_dup_matches.extend(
                self.dup_index.find_matches(image_path, exclude_listing_id=bundle.listing_id)
            )
            if register_images:
                self.dup_index.add(
                    image_id=f"{bundle.listing_id}_{i}",
                    image_path=image_path,
                    listing_id=bundle.listing_id,
                    agent_id=bundle.agent_id,
                )

        return evaluate(
            listing_id=bundle.listing_id,
            permit_check=permit_result,
            watermark_detections=all_watermark_detections,
            duplicate_matches=all_dup_matches,
        )