"""Bootstrap a watermark/logo detection set from Pexels interiors + synthetic marks."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from training.synthesize import overlay_logo, write_yolo_label


def build_dataset(source: Path, dest: Path, seed: int = 42) -> int:
    rng = random.Random(seed)
    images = sorted(
        p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        raise SystemExit(f"No source photos in {source}. Run training/fetch_pexels.py first.")
    img_dir = dest / "images"
    lbl_dir = dest / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        out, box = overlay_logo(bgr, seed=rng.randint(0, 10_000))
        stem = f"{path.stem}_wm"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), out)
        write_yolo_label(lbl_dir / f"{stem}.txt", box)
        count += 1
    (dest / "classes.txt").write_text("watermark_logo\n")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/pexels"))
    parser.add_argument("--dest", type=Path, default=Path("data/yolo"))
    args = parser.parse_args()
    n = build_dataset(args.source, args.dest)
    print(f"Wrote {n} labelled overlays to {args.dest}")


if __name__ == "__main__":
    main()
