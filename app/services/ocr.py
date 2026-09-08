"""OCR normalisation and candidate extraction.

Azure AI Vision is used when credentials are present; otherwise Tesseract.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

CONFUSABLES = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
    }
)

_DIGIT_RUN = re.compile(r"\d[\d\s\-_.]{6,14}\d")
_PHONE = re.compile(r"(?:\+?971|00\s*971|0)?[\s\-]*5\d[\s\-]*\d{3}[\s\-]*\d{4}")
_PRICE = re.compile(r"(?:AED|DHS|\u062f\.\u0625)\s*[\d,]+", re.IGNORECASE)
_PERMIT_HINT = re.compile(
    r"(?:permit|trakheesi|\u062a\u0631\u062e\u064a\u0635|\u0631\u0642\u0645\s*\u0627\u0644\u062a\u0635\u0631\u064a\u062d|ad\s*permit)\s*[:#]?\s*([\d\- ]{8,16})",
    re.IGNORECASE,
)


def fold_confusables(token: str) -> str:
    return token.translate(CONFUSABLES)


def normalize_ocr_text(text: str) -> str:
    collapsed = re.sub(r"[^\S\n]+", " ", text)
    collapsed = collapsed.replace("\u2013", "-").replace("\u2014", "-")
    return collapsed.strip()


def strip_phones_and_prices(text: str) -> str:
    text = _PHONE.sub(" ", text)
    text = _PRICE.sub(" ", text)
    text = re.sub(r"\bAED\s*", " ", text, flags=re.IGNORECASE)
    return text


def extract_permit_candidates(ocr_text: str) -> list[str]:
    """Field-aware extraction: drop phones/prices, keep permit-shaped digit runs."""
    from app.services.permit import is_permit_shape, normalize_permit

    text = normalize_ocr_text(ocr_text)
    hinted: list[str] = []
    for match in _PERMIT_HINT.finditer(text):
        hinted.append(normalize_permit(fold_confusables(match.group(1))))

    remainder = strip_phones_and_prices(text)
    remainder = fold_confusables(remainder)
    raw_runs = [normalize_permit(m.group(0)) for m in _DIGIT_RUN.finditer(remainder)]

    ordered: list[str] = []
    for token in hinted + raw_runs:
        if is_permit_shape(token) and token not in ordered:
            ordered.append(token)
    return ordered


def tesseract_ocr(bgr: np.ndarray) -> str:
    import pytesseract

    try:
        return pytesseract.image_to_string(bgr, lang="eng+ara")
    except Exception:
        return pytesseract.image_to_string(bgr, lang="eng")


def azure_vision_ocr(bgr: np.ndarray, settings: Settings) -> str:
    """Read text via Azure AI Vision Image Analysis 4.0 REST API."""
    import cv2
    import httpx

    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for Azure Vision")
    url = settings.azure_vision_endpoint.rstrip("/") + "/computervision/imageanalysis:analyze"
    params = {"api-version": "2024-02-01", "features": "read"}
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_vision_key,
        "Content-Type": "application/octet-stream",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, params=params, headers=headers, content=encoded.tobytes())
        response.raise_for_status()
    payload = response.json()
    lines: list[str] = []
    blocks = payload.get("readResult", {}).get("blocks", [])
    for block in blocks:
        for line in block.get("lines", []):
            if "text" in line:
                lines.append(line["text"])
    return "\n".join(lines)


def read_text(bgr: np.ndarray, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if settings.azure_vision_enabled:
        try:
            return azure_vision_ocr(bgr, settings)
        except Exception:
            logger.exception("Azure AI Vision OCR failed; falling back to Tesseract")
    try:
        return tesseract_ocr(bgr)
    except Exception:
        logger.exception("Tesseract OCR failed")
        return ""


OcrFn = Callable[[np.ndarray], str]
