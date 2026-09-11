import glob
import os

import pytest

from src.watermark_detector import WatermarkDetector

MODEL_PATH = "models/watermark_yolov8n.onnx"
VAL_DIR = "data/synthetic/watermark_yolo"

pytestmark = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH), reason="trained model not present — run train_watermark_yolo.py first"
)


@pytest.fixture(scope="module")
def detector():
    # imgsz omitted deliberately — auto-detected from the model file itself,
    # so this test doesn't break every time the model gets retrained at a
    # different --imgsz (this is exactly the bug that motivated the change:
    # a hardcoded 256 here broke when the model was retrained at 416).
    # conf_threshold omitted too — uses WatermarkDetector's own default
    # (0.10), chosen by sweeping this exact val set. See that class's
    # docstring for why 0.10 and not YOLO's usual 0.25.
    return WatermarkDetector(MODEL_PATH)


def test_detects_watermark_on_positive_val_images(detector):
    label_dir = f"{VAL_DIR}/labels/val"
    img_dir = f"{VAL_DIR}/images/val"
    if not os.path.exists(label_dir):
        pytest.skip("synthetic val set not generated")

    correct = 0
    total = 0
    for label_path in sorted(glob.glob(f"{label_dir}/*.txt")):
        name = os.path.basename(label_path).replace(".txt", "")
        img_path = f"{img_dir}/{name}.jpg"
        has_watermark_gt = os.path.getsize(label_path) > 0
        predicted = detector.has_watermark(img_path)
        correct += int(predicted == has_watermark_gt)
        total += 1

    accuracy = correct / total
    assert accuracy >= 0.85  # trained model should clear this comfortably