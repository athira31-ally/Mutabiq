from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.models.schemas import (
    BoundingBox,
    CheckResult,
    ComplianceReport,
    ImageFinding,
    ListingMetadata,
)
from app.services.duplicates import DuplicateIndex, phash_hex
from app.services.fuzzy import reconcile
from app.services.ocr import read_text
from app.services.permit import validate_permit_metadata
from app.services.watermark import load_detector, watermark_check

OcrFn = Callable[[np.ndarray], str]
DetectFn = Callable[[np.ndarray], list[BoundingBox]]

MODEL_CARD = {
    "architecture": "YOLOv8n",
    "export": "ONNX",
    "runtime": "onnxruntime",
    "class": "watermark_logo",
    "map50": 0.908,
    "precision": 0.966,
    "recall": 0.824,
    "held_out_accuracy_before_sweep": 0.80,
    "held_out_accuracy_after_sweep": 0.85,
    "conf_threshold": 0.42,
}


class CompliancePipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ocr_fn: OcrFn | None = None,
        detect_fn: DetectFn | None = None,
        duplicate_index: DuplicateIndex | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ocr_fn = ocr_fn or (lambda img: read_text(img, self.settings))
        detector = None if detect_fn else load_detector(self.settings)
        self.detect_fn = detect_fn or (
            (lambda img: detector.detect(img)) if detector else (lambda img: [])
        )
        self.duplicate_index = duplicate_index or DuplicateIndex(
            self.settings.duplicate_index_path,
            threshold=self.settings.phash_hamming_threshold,
        )
        MODEL_CARD["conf_threshold"] = self.settings.detect_conf_threshold

    def inspect(
        self,
        listing: ListingMetadata,
        images: Sequence[tuple[str, np.ndarray]],
    ) -> ComplianceReport:
        checks: list[CheckResult] = [validate_permit_metadata(listing)]
        findings: list[ImageFinding] = []
        ocr_chunks: list[str] = []
        watermark_boxes: list[BoundingBox] = []
        dup_checks: list[CheckResult] = []

        for filename, bgr in images:
            if bgr is None or bgr.size == 0:
                continue
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            text = self.ocr_fn(bgr)
            ocr_chunks.append(text)
            boxes = self.detect_fn(bgr)
            watermark_boxes.extend(boxes)
            digest = phash_hex(bgr)
            dup_checks.append(self.duplicate_index.check(digest, listing.listing_id))
            self.duplicate_index.add(digest, listing.listing_id)
            from app.services.ocr import extract_permit_candidates

            findings.append(
                ImageFinding(
                    filename=filename,
                    watermarks=boxes,
                    ocr_text=text,
                    permit_candidates=extract_permit_candidates(text),
                    phash=digest,
                )
            )

        self.duplicate_index.persist()
        combined_ocr = "\n".join(ocr_chunks)
        checks.append(reconcile(combined_ocr, listing))
        checks.append(watermark_check(watermark_boxes))
        if dup_checks:
            worst = min(dup_checks, key=lambda c: c.score)
            checks.append(worst)
        else:
            checks.append(
                CheckResult(
                    name="duplicate_photo",
                    passed=False,
                    score=0.0,
                    detail="No listing photos were supplied.",
                    evidence={},
                )
            )

        passed = all(c.passed for c in checks)
        overall = float(np.mean([c.score for c in checks])) if checks else 0.0
        return ComplianceReport(
            listing_id=listing.listing_id,
            compliant=passed,
            overall_score=round(overall, 4),
            checks=checks,
            images=findings,
            model=dict(MODEL_CARD),
        )


def decode_image(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
