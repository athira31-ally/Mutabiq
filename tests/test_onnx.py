import numpy as np

from app.inference.onnx_engine import OnnxDetector
from app.models.schemas import BoundingBox


def test_engine_reads_input_hw_from_model(dummy_onnx) -> None:
    engine = OnnxDetector(dummy_onnx, conf_threshold=0.42)
    assert engine.input_height == 320
    assert engine.input_width == 320
    assert engine.input_name == "images"


def test_confidence_threshold_filters(dummy_onnx) -> None:
    engine = OnnxDetector(dummy_onnx, conf_threshold=0.42)
    # Layout after transpose in _decode: rows are anchors, cols = cx,cy,w,h,conf
    raw = np.zeros((1, 5, 2), dtype=np.float32)
    raw[0, :, 0] = [160, 160, 40, 40, 0.91]
    raw[0, :, 1] = [80, 80, 20, 20, 0.10]
    boxes = engine._decode(raw, scale=1.0, h0=320, w0=320)
    assert len(boxes) == 1
    assert isinstance(boxes[0], BoundingBox)
    assert boxes[0].confidence >= 0.42
