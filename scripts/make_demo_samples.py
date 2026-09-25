"""Build the demo listing images served by the live app (demo/samples/).

    python -m scripts.make_demo_samples

Each sample is a real Pexels property photo (from data/raw/stock_photos, Pexels License) with, where the
scenario needs it, a permit banner and/or a synthetic brokerage wordmark composited on top - the same kind
of overlay the watermark detector was trained on. The images are committed, so the Docker image needs no
fonts and nothing is generated at runtime.

Scenarios (the expected verdict is checked by tests/test_demo_samples.py):
  a_compliant   permit banner matches the ad's claimed permit           -> pass
  b_watermark   same, plus an unauthorised brokerage wordmark            -> review (WATERMARK_DETECTED)
  c_mismatch    banner permit differs from the claimed permit            -> review (PERMIT_MISMATCH)
  d_no_permit   no permit anywhere in the image                          -> fail   (PERMIT_MISSING)
  e_reused      sample A's photo re-posted by another agent (re-cropped) -> review (DUPLICATE_PHOTO)
  f_qr_permit   no printed number, only a permit QR code (the portal style) -> pass

The QR in f_qr_permit is generated here and points at a placeholder domain; its link has the same shape as
a real portal permit QR (…/api/listing/<id>/permitValidation/<signature>), see src/qr_permit.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.synthetic_watermark_data import IMG_SIZE, _overlay_logo

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "data" / "raw" / "stock_photos"
OUT = ROOT / "demo" / "samples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def training_style_logo(text: str, seed: int = 7) -> Image.Image:
    """The wordmark exactly as the committed model saw it in training. The training data was generated on
    macOS, where _make_logo's DejaVu font path doesn't exist, so every training logo fell back to Pillow's
    small default font. (Known limitation: the model doesn't yet generalise to large bold logos - see README.)"""
    font = ImageFont.load_default()
    tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    logo = Image.new("RGBA", (x1 - x0 + 20, y1 - y0 + 20), (0, 0, 0, 0))
    ImageDraw.Draw(logo).text((10, 10), text, font=font, fill=(255, 255, 255, 255))
    return logo


SCENARIOS = [
    {"id": "a_compliant", "photo": "luxury_living_room_14613702.jpg", "banner": "7169578165",
     "claimed": "7169578165", "listing_id": "DEMO-A", "agent_id": "AGENT-101",
     "title": "Compliant listing", "expect": "pass"},
    {"id": "b_watermark", "photo": "minimalist_living_room_12277020.jpg", "banner": "6045128830",
     "claimed": "6045128830", "listing_id": "DEMO-B", "agent_id": "AGENT-102", "logo": "SKYLINE ESTATES",
     "title": "Unauthorised watermark", "expect": "review"},
    {"id": "c_mismatch", "photo": "dubai_apartment_interior_14180360.jpg", "banner": "1239982634",
     "claimed": "7169578165", "listing_id": "DEMO-C", "agent_id": "AGENT-103",
     "title": "Permit doesn't match the ad", "expect": "review"},
    {"id": "d_no_permit", "photo": "empty_apartment_interior_11861450.jpg", "banner": None,
     "claimed": "5501234567", "listing_id": "DEMO-D", "agent_id": "AGENT-104",
     "title": "No permit on the listing", "expect": "fail"},
    {"id": "e_reused", "photo": "luxury_living_room_14613702.jpg", "banner": "7169578165",
     "claimed": "7169578165", "listing_id": "DEMO-E", "agent_id": "AGENT-205", "crop": 0.96,
     "title": "Photo reused by another agent", "expect": "review"},
    {"id": "f_qr_permit", "photo": "furnished_apartment_bedroom_27604135.jpg", "banner": None,
     "qr": "https://portal.example.ae/api/listing/DEMO-F/permitValidation/"
           "MEUCIQDemoOnlyNotARealSignature0000000000000000000AiEAdemo0000000000000000000000000000000",
     "claimed": "", "listing_id": "DEMO-F", "agent_id": "AGENT-106",
     "title": "Permit shown as a QR code", "expect": "pass"},
]


def square(img: Image.Image, crop: float = 1.0) -> Image.Image:
    w, h = img.size
    s = int(min(w, h) * crop)    # crop < 1: a slightly tighter crop, like a re-post of the same photo
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((IMG_SIZE, IMG_SIZE))


def add_banner(img: Image.Image, permit: str) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    h = 58
    d.rectangle([0, IMG_SIZE - h, IMG_SIZE, IMG_SIZE], fill=(255, 255, 255))
    font = ImageFont.truetype(FONT, 24)
    d.text((18, IMG_SIZE - h + 15), f"Trakheesi Permit No. {permit}", font=font, fill=(20, 20, 20))
    return img


def add_permit_qr(img: Image.Image, link: str) -> Image.Image:
    """A white permit card with the QR in the bottom-right corner. (No caption text: the watermark model, trained
    on text wordmarks, flags a 'Trakheesi Permit' caption as a logo - see README, 'What going live taught me'.)"""
    import zxingcpp
    qr = Image.fromarray(np.array(zxingcpp.create_barcode(link, zxingcpp.BarcodeFormat.QRCode).to_image(scale=4)))
    qr = qr.convert("RGB").resize((150, 150), Image.NEAREST)
    img = img.copy()
    x0, y0 = IMG_SIZE - 150 - 34, IMG_SIZE - 150 - 58
    d = ImageDraw.Draw(img)
    d.rectangle([x0 - 12, y0 - 12, IMG_SIZE - 22, IMG_SIZE - 46], fill=(255, 255, 255))
    img.paste(qr, (x0, y0))
    return img


def build() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for s in SCENARIOS:
        img = square(Image.open(PHOTOS / s["photo"]).convert("RGB"), crop=s.get("crop", 1.0))
        if s.get("logo"):
            img, _ = _overlay_logo(img, training_style_logo(s["logo"]), random.Random(5))
        if s["banner"]:
            img = add_banner(img, s["banner"])
        if s.get("qr"):
            img = add_permit_qr(img, s["qr"])
        img.save(OUT / f"{s['id']}.jpg", quality=90)
        meta.append({k: s[k] for k in ("id", "title", "claimed", "listing_id", "agent_id", "expect")}
                    | {"image": f"{s['id']}.jpg"})
    (OUT / "samples.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    for m in build():
        print(m["id"], "->", m["expect"])
