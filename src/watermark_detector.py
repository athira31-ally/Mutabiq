"""ONNX Runtime inference for the watermark/logo detector.

Deliberately doesn't depend on ultralytics at inference time — only
onnxruntime + numpy + Pillow, the same lean footprint as the SmartPoseEdge
deployment. Training (train_watermark_yolo.py) uses ultralytics; serving
doesn't need to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "watermark"


def _letterbox(img: Image.Image, size: int = 320) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + pad to a square `size x size` input, preserving aspect ratio
    (standard YOLO preprocessing). Returns (chw_float32_array, scale, (pad_x, pad_y)).
    """
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
    return chw, scale, (pad_x, pad_y)


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    xc, yc, w, h = box
    return np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]

        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-9)

        order = rest[iou <= iou_threshold]
    return keep


class WatermarkDetector:
    """Wraps the trained watermark-detection ONNX model behind a simple
    `.detect(image_path)` call.

    conf_threshold default (0.10, not YOLO's usual 0.25): chosen by sweeping
    the real-photo-trained model's confidence threshold against the
    validation set (0.25/0.20 -> 80% pass/fail accuracy, 0.10 -> 85%, no
    further gain below that). Deliberately favors recall over precision —
    ARCHITECTURE.md's evaluation design calls this out explicitly: a missed
    watermark (false negative) is the costlier failure than an extra false
    alarm, since a false alarm here only produces a "review" flag, not a
    hard fail.
    """

    def __init__(
        self, onnx_path: str | Path, imgsz: int | None = None, conf_threshold: float = 0.10
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        if imgsz is None:
            # Read the model's own expected input size instead of hardcoding
            # a number that has to be kept in sync with whatever --imgsz the
            # model happened to be trained/exported at. ONNX input shape is
            # [batch, channels, height, width]; height/width come back as
            # plain ints for a fixed-size export (what export_onnx() here
            # produces), or as a symbolic name (a string) for a dynamic-shape
            # export — fall back to YOLO's own conventional default (640) in
            # that case, since there's nothing fixed to read.
            input_shape = self.session.get_inputs()[0].shape
            height = input_shape[2]
            imgsz = height if isinstance(height, int) else 640

        self.imgsz = imgsz
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str) -> list[Detection]:
        img = Image.open(image_path).convert("RGB")
        chw, scale, (pad_x, pad_y) = _letterbox(img, self.imgsz)

        outputs = self.session.run(None, {self.input_name: chw.astype(np.float32)})
        pred = outputs[0][0]  # (5, N) for a single-class model: [xc, yc, w, h, conf]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # normalize to (N, 5)

        boxes_xyxy, scores = [], []
        for row in pred:
            conf = float(row[4])
            if conf < self.conf_threshold:
                continue
            xyxy = _xywh_to_xyxy(row[:4])
            boxes_xyxy.append(xyxy)
            scores.append(conf)

        if not boxes_xyxy:
            return []

        boxes_xyxy = np.array(boxes_xyxy)
        scores = np.array(scores)
        keep = _nms(boxes_xyxy, scores)

        detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            # undo letterbox padding + scale to map back to original image coords
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            detections.append(
                Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                           confidence=float(scores[idx]))
            )
        return detections

    def has_watermark(self, image_path: str) -> bool:
        return len(self.detect(image_path)) > 0