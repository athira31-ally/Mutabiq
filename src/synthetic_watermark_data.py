"""Training data for the watermark/logo detector.

No public dataset of "unauthorized broker watermark on a UAE listing photo"
exists, so this bootstraps one by overlaying a synthetic brokerage logo onto
a base photo at randomized position/scale/opacity, and writing the result
out in YOLO label format. The logo stays synthetic either way — a generated
wordmark, not a real brand's actual logo, which sidesteps trademark
questions a scraped competitor logo would raise.

Two modes for the *base* photo underneath that logo:

  - **Procedural** (default, no setup, works offline): `_make_base_photo()`
    draws a cheap stand-in room — gradient wall, floor band, a few
    furniture-shaped rectangles. Enough visual structure for a logo to sit
    believably on top of, zero external dependencies. This is what the
    first trained model (mAP50 0.995) was built on.
  - **Real photos** (`--real-photos-dir`, recommended once you have some):
    uses actual downloaded interior/apartment photos as the base instead —
    see `fetch_stock_photos.py`, which pulls free-licensed real estate
    photos from the Pexels API. Real backgrounds mean the model learns to
    find a logo against genuinely varied lighting, furniture, and clutter
    instead of a handful of procedural palettes, which is what actually
    matters for real-world precision.

Output layout (YOLO convention):

    <out_dir>/
      images/train/*.jpg   images/val/*.jpg
      labels/train/*.txt   labels/val/*.txt
      dataset.yaml
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

REAL_PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")

IMG_SIZE = 640
CLASS_NAMES = ["watermark"]

BROKERAGE_NAMES = [
    "PRIME KEY REALTY",
    "DXB HOMES",
    "GULF SANDS PROPERTIES",
    "SKYLINE ESTATES",
    "MARINA & CO",
    "PALM REALTY GROUP",
]

ROOM_PALETTES = [
    # (wall, floor, accent)
    ((235, 228, 214), (178, 150, 110), (150, 120, 90)),
    ((220, 224, 226), (140, 120, 100), (90, 90, 100)),
    ((245, 238, 225), (190, 170, 140), (170, 140, 100)),
    ((210, 220, 218), (120, 130, 120), (80, 100, 95)),
]


def _make_base_photo(seed: int) -> Image.Image:
    """A cheap procedural stand-in for a real interior/exterior listing
    photo: a gradient wall, a floor band, and a few furniture-like
    rectangles. Enough visual structure for a watermark to sit believably
    on top of, without needing any external image source.
    """
    rng = random.Random(seed)
    wall, floor, accent = rng.choice(ROOM_PALETTES)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), wall)
    draw = ImageDraw.Draw(img)

    # simple vertical gradient on the wall
    for y in range(IMG_SIZE):
        shade = 1.0 - 0.15 * (y / IMG_SIZE)
        draw.line(
            [(0, y), (IMG_SIZE, y)],
            fill=tuple(int(c * shade) for c in wall),
        )

    # floor band
    floor_top = int(IMG_SIZE * rng.uniform(0.58, 0.7))
    draw.rectangle([0, floor_top, IMG_SIZE, IMG_SIZE], fill=floor)

    # a few "furniture" rectangles
    for _ in range(rng.randint(2, 5)):
        w = rng.randint(60, 180)
        h = rng.randint(60, 200)
        x0 = rng.randint(0, IMG_SIZE - w)
        y0 = rng.randint(int(IMG_SIZE * 0.3), floor_top - 20) if floor_top > 50 else 50
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=accent)

    return img


def _load_real_photo(path: Path, rng: random.Random) -> Image.Image:
    """Turn a real downloaded photo into a base image the same shape the
    rest of the pipeline expects (a square IMG_SIZE x IMG_SIZE crop).

    Resize-then-crop rather than a plain resize: squashing a real photo's
    aspect ratio into a square would visibly distort every room in the
    dataset, which is exactly the kind of artifact a model can latch onto
    instead of learning what actually matters. This resizes so the *short*
    side covers IMG_SIZE, then takes a random square crop out of that —
    real proportions preserved, and randomizing the crop position means
    reusing the same source photo across a few synthetic examples doesn't
    hand the model identical backgrounds every time.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = IMG_SIZE / min(w, h)
    new_w, new_h = max(IMG_SIZE, round(w * scale)), max(IMG_SIZE, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    w, h = img.size
    x0 = rng.randint(0, w - IMG_SIZE) if w > IMG_SIZE else 0
    y0 = rng.randint(0, h - IMG_SIZE) if h > IMG_SIZE else 0
    img = img.crop((x0, y0, x0 + IMG_SIZE, y0 + IMG_SIZE))

    # mild brightness jitter, same reasoning as the crop: keeps repeated
    # source photos from producing visually identical training examples
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))


def _find_real_photos(real_photos_dir: str | Path | None) -> list[Path]:
    if not real_photos_dir:
        return []
    real_photos_dir = Path(real_photos_dir)
    photos = sorted(
        p for p in real_photos_dir.iterdir() if p.suffix.lower() in REAL_PHOTO_EXTENSIONS
    )
    if not photos:
        raise ValueError(
            f"--real-photos-dir was set to {real_photos_dir} but it has no "
            f"{REAL_PHOTO_EXTENSIONS} files in it. Run fetch_stock_photos.py "
            "first, or drop this flag to fall back to procedural base photos."
        )
    return photos


def _make_logo(text: str, seed: int) -> Image.Image:
    """A synthetic wordmark logo, rendered onto a transparent canvas."""
    rng = random.Random(seed)
    font_size = rng.randint(28, 44)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except OSError:
        font = ImageFont.load_default()

    tmp = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20

    logo = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    color = rng.choice([(255, 255, 255), (20, 20, 20), (200, 170, 90)])
    draw.text((10, 10), text, font=font, fill=color + (255,))
    return logo


def _overlay_logo(base: Image.Image, logo: Image.Image, rng: random.Random):
    """Paste `logo` onto `base` at a randomized corner-biased position,
    scale, and opacity. Returns (composited_image, yolo_bbox) where
    yolo_bbox is (x_center, y_center, w, h), all normalized to [0, 1].
    """
    scale = rng.uniform(0.5, 1.1)
    logo = logo.resize((int(logo.width * scale), int(logo.height * scale)))

    opacity = rng.uniform(0.45, 0.95)
    alpha = logo.split()[3].point(lambda p: int(p * opacity))
    logo.putalpha(alpha)

    margin = 12
    corners = [
        (margin, margin),
        (IMG_SIZE - logo.width - margin, margin),
        (margin, IMG_SIZE - logo.height - margin),
        (IMG_SIZE - logo.width - margin, IMG_SIZE - logo.height - margin),
        (
            (IMG_SIZE - logo.width) // 2,
            (IMG_SIZE - logo.height) // 2,
        ),  # occasional dead-center watermark
    ]
    x, y = rng.choice(corners)
    x = max(0, min(x, IMG_SIZE - logo.width))
    y = max(0, min(y, IMG_SIZE - logo.height))

    composited = base.convert("RGBA")
    composited.alpha_composite(logo, (x, y))

    xc = (x + logo.width / 2) / IMG_SIZE
    yc = (y + logo.height / 2) / IMG_SIZE
    w = logo.width / IMG_SIZE
    h = logo.height / IMG_SIZE
    return composited.convert("RGB"), (xc, yc, w, h)


def generate_dataset(
    out_dir: str | Path,
    n_train: int = 60,
    n_val: int = 16,
    positive_ratio: float = 0.75,
    seed: int = 42,
    real_photos_dir: str | Path | None = None,
) -> Path:
    """Generate the full training dataset and write it in YOLO format.
    Returns the path to dataset.yaml.

    With `real_photos_dir` unset, base photos are procedural (fast, no
    setup, matches the original build). With it set to a folder of real
    photos (see `fetch_stock_photos.py`), each example's base photo is a
    random square crop of a real image instead — see `_load_real_photo`.
    """
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    real_photos = _find_real_photos(real_photos_dir)
    if real_photos:
        print(f"Using {len(real_photos)} real base photos from {real_photos_dir}")
    else:
        print("Using procedurally generated base photos (pass --real-photos-dir to use real ones)")

    for split, n in [("train", n_train), ("val", n_val)]:
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            img_seed = rng.randint(0, 1_000_000)
            if real_photos:
                base = _load_real_photo(rng.choice(real_photos), rng)
            else:
                base = _make_base_photo(img_seed)
            is_positive = rng.random() < positive_ratio

            label_lines = []
            if is_positive:
                text = rng.choice(BROKERAGE_NAMES)
                logo = _make_logo(text, img_seed)
                base, (xc, yc, w, h) = _overlay_logo(base, logo, rng)
                label_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            name = f"{split}_{i:04d}"
            base.save(img_dir / f"{name}.jpg", quality=90)
            (lbl_dir / f"{name}.txt").write_text("\n".join(label_lines))

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"""path: {out_dir.resolve()}
train: images/train
val: images/val
names:
  0: watermark
"""
    )
    return yaml_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic/watermark_yolo")
    parser.add_argument("--n-train", type=int, default=60)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument(
        "--real-photos-dir",
        default=None,
        help="Folder of real photos from fetch_stock_photos.py. Omit to use "
        "procedural base photos instead (no setup needed).",
    )
    args = parser.parse_args()

    path = generate_dataset(
        args.out, n_train=args.n_train, n_val=args.n_val, real_photos_dir=args.real_photos_dir
    )
    print(f"Dataset written, config at: {path}")