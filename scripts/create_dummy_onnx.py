"""Build a tiny YOLOv8-shaped ONNX graph for local tests and cold starts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def write_dummy_yolo_onnx(
    path: Path,
    height: int = 320,
    width: int = 320,
    anchors: int = 4,
) -> Path:
    """Constant [1, 5, anchors] output (cx, cy, w, h, conf) of zeros."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, height, width]
    )
    output = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 5, anchors]
    )
    zeros = np.zeros((1, 5, anchors), dtype=np.float32)
    node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=numpy_helper.from_array(zeros),
    )
    graph = helper.make_graph([node], "yolov8n_watermark_dummy", [images], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


if __name__ == "__main__":
    write_dummy_yolo_onnx(Path("models/yolov8n_watermark.onnx"), height=640, width=640)
    print("Wrote models/yolov8n_watermark.onnx")
