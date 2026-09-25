"""End-to-end orchestration: listing bundle in, ComplianceReport out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .dup_hash import DuplicatePhotoIndex
from .ocr_permit import OCRBackend, check_permit, check_permit_text, get_backend
from .qr_permit import PermitQR, read_permit_qrs
from .rule_engine import ComplianceReport, evaluate
from .watermark_detector import WatermarkDetector


@dataclass
class ListingBundle:
    listing_id: str
    agent_id: str
    image_paths: list[str]
    claimed_permit_number: str | None = None
    ad_text: str = ""
    page_text: str = ""                        # text of the listing page (fetched link or uploaded PDF)
    page_qrs: list[PermitQR] = field(default_factory=list)  # QR codes already decoded from that page
    link_listing_ref: str | None = None        # listing ID from the pasted Bayut / Property Finder link


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
        if not bundle.image_paths and not bundle.page_text and not bundle.page_qrs:
            raise ValueError("Listing bundle has no images and no page content.")

        # A printed permit can be on any photo: read them in order and stop at the first that has one.
        permit_result = None
        for image_path in bundle.image_paths:
            result = check_permit(
                image_path, claimed_permit_number=bundle.claimed_permit_number, backend=self.ocr_backend
            )
            permit_result = permit_result or result
            if result.found_numbers:
                permit_result = result
                break
        # ...or in the listing page's own text (the "Regulatory Information" box of a saved page).
        if (permit_result is None or not permit_result.found_numbers) and bundle.page_text:
            from_page = check_permit_text(bundle.page_text, bundle.claimed_permit_number)
            if from_page.found_numbers or permit_result is None:
                permit_result = from_page

        all_watermark_detections = []
        all_dup_matches = []
        permit_qrs = list(bundle.page_qrs)
        for i, image_path in enumerate(bundle.image_paths):
            permit_qrs.extend(read_permit_qrs(image_path))  # the permit QR can be on any image
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
            permit_qrs=permit_qrs,
            link_listing_ref=bundle.link_listing_ref,
        )