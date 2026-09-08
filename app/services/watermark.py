from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.config import Settings, get_settings
from app.inference.onnx_engine import OnnxDetector
from app.models.schemas import BoundingBox, CheckResult

DetectorFn = Callable[[np.ndarray], list[BoundingBox]]


def load_detector(settings: Settings | None = None) -> OnnxDetector | None:
    settings = settings or get_settings()
    if not settings.yolo_onnx_path.exists():
        return None
    return OnnxDetector(settings.yolo_onnx_path, conf_threshold=settings.detect_conf_threshold)


def watermark_check(boxes: list[BoundingBox]) -> CheckResult:
    """Competitor / agency watermarks on listing photos are a RERA advertising risk."""
    if not boxes:
        return CheckResult(
            name="watermark_logo",
            passed=True,
            score=1.0,
            detail="No watermark/logo detections above threshold.",
            evidence={"detections": 0},
        )
    top = max(boxes, key=lambda b: b.confidence)
    return CheckResult(
        name="watermark_logo",
        passed=False,
        score=max(0.0, 1.0 - top.confidence),
        detail=f"{len(boxes)} watermark/logo region(s) detected (top conf {top.confidence:.3f}).",
        evidence={"detections": [b.model_dump() for b in boxes]},
    )
