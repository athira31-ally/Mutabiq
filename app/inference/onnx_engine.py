"""ONNX Runtime inference that reads its input tensor shape from the model file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.models.schemas import BoundingBox


class OnnxDetector:
    """Dependency-light YOLOv8 ONNX runner (no Ultralytics at serve time)."""

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.42) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        self.conf_threshold = conf_threshold
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = tuple(self.session.get_inputs()[0].shape)
        self.input_height, self.input_width = self._spatial_hw(self.input_shape)
        self.output_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def _spatial_hw(shape: tuple) -> tuple[int, int]:
        """Parse NCHW or NHWC, including dynamic dims represented as None/str."""
        dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
        if len(dims) != 4:
            raise ValueError(f"Expected 4D input, got {shape}")
        # NCHW (YOLO export): [1, 3, H, W]
        if dims[1] in (1, 3):
            h = dims[2] or 640
            w = dims[3] or 640
            return h, w
        # NHWC
        h = dims[1] or 640
        w = dims[2] or 640
        return h, w

    def preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h0, w0 = bgr.shape[:2]
        scale = min(self.input_width / w0, self.input_height / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        return tensor, scale, (h0, w0)

    def detect(self, bgr: np.ndarray) -> list[BoundingBox]:
        tensor, scale, (h0, w0) = self.preprocess(bgr)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        return self._decode(outputs[0], scale, h0, w0)

    def _decode(self, raw: np.ndarray, scale: float, h0: int, w0: int) -> list[BoundingBox]:
        """Decode YOLOv8 export layout [1, 4+nc, n] or a dummy [1, 6, n] tensor."""
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            return []
        # YOLOv8 export is (4+nc, anchors). Tiny test graphs may have few anchors.
        if pred.shape[0] in (4, 5, 6, 84) or pred.shape[0] < pred.shape[1]:
            pred = pred.T
        boxes: list[BoundingBox] = []
        for row in pred:
            if row.size < 5:
                continue
            cx, cy, w, h = row[:4]
            scores = row[4:]
            conf = float(np.max(scores))
            if conf < self.conf_threshold:
                continue
            x1 = (cx - w / 2) / scale
            y1 = (cy - h / 2) / scale
            x2 = (cx + w / 2) / scale
            y2 = (cy + h / 2) / scale
            boxes.append(
                BoundingBox(
                    x1=float(np.clip(x1, 0, w0)),
                    y1=float(np.clip(y1, 0, h0)),
                    x2=float(np.clip(x2, 0, w0)),
                    y2=float(np.clip(y2, 0, h0)),
                    confidence=conf,
                    label="watermark_logo",
                )
            )
        return _nms(boxes, iou_threshold=0.45)


def _nms(boxes: list[BoundingBox], iou_threshold: float) -> list[BoundingBox]:
    if not boxes:
        return []
    order = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep: list[BoundingBox] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [b for b in order if _iou(best, b) < iou_threshold]
    return keep


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0
