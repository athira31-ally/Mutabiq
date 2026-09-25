"""Is a detected watermark the listing agency's own logo?

Agencies brand their own photos, and that is allowed. The violation is another broker's mark on the ad
(usually a copied photo). Found on a real Bayut listing: SEROVIA PROPERTIES' photos carry a gold SEROVIA
logo, which the detector rightly finds but which must not count against SEROVIA's own listing.

For each detected mark we crop around it, read it with the same OCR backend the pipeline uses (Azure AI
Vision when configured, else Tesseract), and fuzzy-match the text against the *distinctive* words of the
registered agency name - "SEROVIA", not "PROPERTIES" or "L.L.C", which every agency shares. Stylised logos
read imperfectly (the "S" of SEROVIA is part of its building icon, so OCR sees "EROVIA"), hence the fuzzy,
window-based match. A mark we can't read, or that reads as something else, stays a violation.
"""
from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image, ImageOps

GENERIC_WORDS = {
    "PROPERTIES", "PROPERTY", "REAL", "ESTATE", "REALTY", "REALTORS", "BROKERAGE", "BROKERS", "BROKER",
    "GROUP", "HOMES", "HOME", "LIVING", "INVESTMENT", "INVESTMENTS", "DEVELOPMENTS", "CAPITAL", "INTERNATIONAL",
    "LLC", "L.L.C", "LTD", "FZ", "FZE", "FZCO", "CO", "AND", "THE", "OF", "DUBAI", "UAE", "EMIRATES", "MIDDLE", "EAST",
}
MATCH_THRESHOLD = 0.75
CROPS = ((0.5, 1.0), (1.5, 3.0))       # tight crop first, then wide (a detection often covers part of a logo)


@dataclass
class BrandCheck:
    own: bool
    text: str
    similarity: float
    token: str | None


def distinctive_tokens(agency: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9.&]+", agency.upper())
    return [w for w in words if w.strip(".") not in GENERIC_WORDS and w not in GENERIC_WORDS and len(w) >= 4]


def brand_similarity(ocr_text: str, agency: str) -> tuple[float, str | None]:
    """Best fuzzy match of any distinctive agency word inside the OCR text (letters only, spaces ignored)."""
    letters = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())
    best, best_token = 0.0, None
    for token in distinctive_tokens(agency):
        t = re.sub(r"[^A-Z0-9]", "", token)
        if not letters:
            break
        for size in {len(t) - 1, len(t), len(t) + 1}:
            if size <= 0:
                continue
            for i in range(0, max(1, len(letters) - size + 1)):
                r = SequenceMatcher(None, t, letters[i:i + size]).ratio()
                if r > best:
                    best, best_token = r, token
    return best, best_token


def _variants(crop: Image.Image, tesseract: bool):
    if not tesseract:              # Azure AI Vision reads colour, low-contrast and stylised text as it is
        yield crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS) if crop.width < 600 else crop
        return
    big = crop.resize((crop.width * 4, crop.height * 4), Image.BICUBIC)
    gray = ImageOps.grayscale(big)
    yield ImageOps.autocontrast(gray)
    yield ImageOps.invert(gray)


def _read(img: Image.Image, ocr_backend, path: Path) -> str:
    from .ocr_permit import TesseractOCR
    if isinstance(ocr_backend, TesseractOCR):
        # A logo is one short block of text: single-block mode (--psm 6) reads it far better than page
        # layout mode. Still best effort on stylised logos - on the real SEROVIA mark only a few settings
        # read "ROVIA"; Azure AI Vision is the reader this check is meant for.
        import pytesseract
        return pytesseract.image_to_string(img, config="--psm 6")
    img.save(path)
    return ocr_backend.extract_text(str(path))


def check_detection(image_path: str, det, agency: str, ocr_backend) -> BrandCheck:
    from .ocr_permit import TesseractOCR
    if not distinctive_tokens(agency):
        return BrandCheck(False, "", 0.0, None)
    texts = []
    with Image.open(image_path) as img, tempfile.TemporaryDirectory() as tmp:
        img = img.convert("RGB")
        bw, bh = det.x2 - det.x1, det.y2 - det.y1
        for mx, my in CROPS:
            box = (max(0, det.x1 - bw * mx), max(0, det.y1 - bh * my),
                   min(img.width, det.x2 + bw * mx), min(img.height, det.y2 + bh * my))
            crop = img.crop(tuple(int(v) for v in box))
            for variant in _variants(crop, isinstance(ocr_backend, TesseractOCR)):
                text = _read(variant, ocr_backend, Path(tmp) / f"mark_{len(texts)}.png")
                texts.append(text)
                score, token = brand_similarity(text, agency)
                if score >= MATCH_THRESHOLD:
                    return BrandCheck(True, " ".join(text.split())[:80], round(score, 2), token)
    joined = " | ".join(" ".join(t.split())[:40] for t in texts if t.strip())
    score, token = brand_similarity(" ".join(texts), agency)
    return BrandCheck(False, joined[:80], round(score, 2), token)
