"""Fine-tune YOLOv8n on the synthetic watermark set, then export ONNX.

Runtime serving does not import Ultralytics; only this training job does.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _write_data_yaml(root: Path) -> Path:
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                "train: images",
                "val: images",
                "names:",
                "  0: watermark_logo",
                "",
            ]
        )
    )
    return yaml_path


def train_and_export(
    data_root: Path,
    out_dir: Path,
    epochs: int = 40,
    imgsz: int = 640,
) -> Path:
    from ultralytics import YOLO

    data_yaml = _write_data_yaml(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=8,
        project=str(out_dir),
        name="watermark",
        exist_ok=True,
    )
    best = out_dir / "watermark" / "weights" / "best.pt"
    exported = YOLO(str(best)).export(format="onnx", dynamic=False, simplify=True)
    onnx_path = Path(exported)
    target = Path("models/yolov8n_watermark.onnx")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(onnx_path.read_bytes())
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/yolo"))
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    path = train_and_export(args.data, args.out, epochs=args.epochs)
    print(f"Exported {path}")


if __name__ == "__main__":
    main()
