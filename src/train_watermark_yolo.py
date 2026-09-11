"""Fine-tune YOLOv8n to detect watermark/logo overlays on listing photos,
then export to ONNX.

Same inference path as SmartPoseEdge (YOLO -> ONNX -> ONNX Runtime), pointed
at a compliance signal instead of a pose keypoint.

Usage:
    python -m src.train_watermark_yolo --data data/synthetic/watermark_yolo/dataset.yaml \\
        --epochs 3 --imgsz 320 --out models/

The defaults below (few epochs, small image size) are a *smoke test* over
the synthetic dataset — enough to prove the training -> export -> inference
path is wired correctly end to end, on CPU, in a couple of minutes. Real
performance numbers need a real logo/watermark dataset and a proper training
budget (more epochs, larger imgsz, a GPU) — see ARCHITECTURE.md.

NOTE on `out_dir`: `data_yaml` and `out_dir` are both resolved to absolute
paths before being handed to ultralytics. Reason: ultralytics keeps a
one-time global settings file (`~/Library/Application Support/Ultralytics/
settings.json` on macOS) that records a `runs_dir` the *first* time the
package is ever used on a machine — and a relative `project=` path gets
anchored under that stored global directory, not under wherever you're
currently running the command from. If ultralytics was ever used from a
different project on this machine before, a relative `--out models` here
silently lands inside *that* project's folder instead of this one. Absolute
paths sidestep the ambiguity entirely regardless of what that global
setting happens to be.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train(
    data_yaml: str,
    epochs: int = 3,
    imgsz: int = 320,
    out_dir: str = "models",
    model_size: str = "yolov8n.pt",
) -> Path:
    from ultralytics import YOLO

    data_yaml = str(Path(data_yaml).resolve())
    out_dir = str(Path(out_dir).resolve())

    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=out_dir,
        name="watermark_detector",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    return best_pt


def export_onnx(weights_path: str | Path, imgsz: int = 320) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    return Path(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic/watermark_yolo/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    best = train(args.data, epochs=args.epochs, imgsz=args.imgsz, out_dir=args.out)
    print(f"Best weights: {best}")

    onnx_path = export_onnx(best, imgsz=args.imgsz)
    print(f"ONNX export: {onnx_path}")