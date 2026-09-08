from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Geometric agency-mark templates. Intentionally not real brokerage wordmarks.
GENERIC_MARKS = ("MARK", "AGCY", "LIST", "PROP", "VIEW", "HOLD")


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_generic_logo(size: int = 220, seed: int | None = None) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (rng.randint(20, 80), rng.randint(20, 80), rng.randint(90, 180), rng.randint(160, 220))
    inset = size // 10
    shape = rng.choice(["circle", "rounded", "bars"])
    if shape == "circle":
        draw.ellipse([inset, inset, size - inset, size - inset], outline=color, width=size // 18)
    elif shape == "rounded":
        draw.rounded_rectangle(
            [inset, inset, size - inset, size - inset],
            radius=size // 8,
            outline=color,
            width=size // 18,
        )
    else:
        for i in range(3):
            y = inset + i * (size // 4)
            draw.rectangle([inset, y, size - inset, y + size // 10], fill=color)
    text = rng.choice(GENERIC_MARKS)
    font = _font(size // 5)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2, (size - th) / 2), text, fill=color, font=font)
    return img


def overlay_logo(
    base_bgr: np.ndarray,
    logo: Image.Image | None = None,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Paste a synthetic logo and return YOLO-normalized cx, cy, w, h."""
    rng = random.Random(seed)
    h, w = base_bgr.shape[:2]
    logo = logo or render_generic_logo(seed=seed)
    scale = rng.uniform(0.08, 0.18)
    lw = max(24, int(w * scale))
    lh = max(24, int(lw * logo.size[1] / logo.size[0]))
    logo = logo.resize((lw, lh), Image.Resampling.LANCZOS)
    x = rng.randint(0, max(0, w - lw))
    y = rng.randint(0, max(0, h - lh))
    base = Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    base.paste(logo, (x, y), logo)
    out = cv2.cvtColor(np.array(base.convert("RGB")), cv2.COLOR_RGB2BGR)
    cx = (x + lw / 2) / w
    cy = (y + lh / 2) / h
    return out, (cx, cy, lw / w, lh / h)


def write_yolo_label(path: Path, yolo_box: tuple[float, float, float, float], class_id: int = 0) -> None:
    cx, cy, bw, bh = yolo_box
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
