"""Permit OCR extraction and validation.

Pulls text off a listing image (permit numbers are usually overlaid on the
photo itself, not just in the ad copy), matches it against the expected
Trakheesi (Dubai) / Madhmoun (Abu Dhabi) permit-number shape, and cross-checks
it against whatever the listing metadata *claims* the permit number is.

Three OCR backends share one interface so the rest of the pipeline never has
to know which one is running:

  - AzureVisionOCR   — production backend, calls Azure AI Vision's Read API.
  - TesseractOCR     — local/offline backend, no cloud dependency or API key.
  - MockOCR          — deterministic backend for tests (returns text you hand it).

NOTE on the permit format: confirmed against DLD's own Trakheesi validation
page (dubailand.gov.ae) — a Trakheesi permit number is a continuous numeric
string, 8-12 digits, no letter prefix (e.g. "1239982634", not "DLD-1239982634").
Modern listings also carry a companion Madmoun QR code next to the number,
which is out of scope here (decoding a QR code is a different, much easier
CV problem than watermark detection — a natural follow-up, not built yet).
PERMIT_REGEX below matches that confirmed format. One real ambiguity worth
knowing about: a bare 8-12 digit run can also match a phone number sitting
elsewhere in the ad text (a UAE mobile number is typically 9-12 digits
depending on how it's formatted) — this regex doesn't try to disambiguate by
nearby keywords like "Permit #", so a false-positive extraction from a phone
number is possible on ad text that includes one. Worth a keyword-proximity
check as a future improvement; not needed for the cross-check logic below,
which just compares whatever it found against the claimed number.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum


# Confirmed against DLD's Trakheesi validation page: a continuous 8-12 digit
# string, no letter prefix. See the module docstring for the phone-number
# ambiguity this leaves open.
PERMIT_REGEX = re.compile(r"\b\d{8,12}\b")

EMIRATE_SYSTEMS = {
    "dubai": "Trakheesi",
    "abu_dhabi": "Madhmoun",
}


class Emirate(str, Enum):
    DUBAI = "dubai"
    ABU_DHABI = "abu_dhabi"


# --------------------------------------------------------------------------
# OCR backends
# --------------------------------------------------------------------------

class OCRBackend(ABC):
    """Common interface every OCR backend implements."""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Return all text found in the image, newline-separated."""


class MockOCR(OCRBackend):
    """Deterministic backend for tests — hand it the text you want back."""

    def __init__(self, canned_text: str = ""):
        self.canned_text = canned_text

    def extract_text(self, image_path: str) -> str:
        return self.canned_text


class TesseractOCR(OCRBackend):
    """Local OCR via pytesseract. No API key, no network call — the fallback
    backend for offline dev and for anyone running this without Azure access.
    """

    def __init__(self):
        try:
            import pytesseract  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "pytesseract is not installed. `pip install pytesseract` "
                "and make sure the `tesseract` binary is on PATH."
            ) from e

    def extract_text(self, image_path: str) -> str:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as img:
            return pytesseract.image_to_string(img)


class AzureVisionOCR(OCRBackend):
    """Production backend — Azure AI Vision's Read API.

    Requires AZURE_VISION_ENDPOINT and AZURE_VISION_KEY. Not exercised in
    this repo's test suite (no live Azure resource here); TesseractOCR and
    MockOCR cover the same `extract_text` contract so the rest of the
    pipeline is identical either way.
    """

    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self.endpoint = endpoint or os.environ.get("AZURE_VISION_ENDPOINT")
        self.key = key or os.environ.get("AZURE_VISION_KEY")
        if not self.endpoint or not self.key:
            raise RuntimeError(
                "AzureVisionOCR needs AZURE_VISION_ENDPOINT and AZURE_VISION_KEY "
                "(env vars or constructor args)."
            )

    def extract_text(self, image_path: str) -> str:
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.ai.vision.imageanalysis.models import VisualFeatures
        from azure.core.credentials import AzureKeyCredential

        client = ImageAnalysisClient(self.endpoint, AzureKeyCredential(self.key))
        with open(image_path, "rb") as f:
            result = client.analyze(
                image_data=f.read(), visual_features=[VisualFeatures.READ]
            )
        if not result.read:
            return ""
        lines = []
        for block in result.read.blocks:
            for line in block.lines:
                lines.append(line.text)
        return "\n".join(lines)


def get_backend(name: str = "auto", **kwargs) -> OCRBackend:
    """Factory: 'azure', 'tesseract', 'mock', or 'auto' (Azure if credentials
    are set, else Tesseract)."""
    if name == "mock":
        return MockOCR(**kwargs)
    if name == "azure":
        return AzureVisionOCR(**kwargs)
    if name == "tesseract":
        return TesseractOCR()
    if name == "auto":
        if os.environ.get("AZURE_VISION_KEY"):
            return AzureVisionOCR(**kwargs)
        return TesseractOCR()
    raise ValueError(f"Unknown OCR backend: {name}")


# --------------------------------------------------------------------------
# Permit extraction + validation
# --------------------------------------------------------------------------

@dataclass
class PermitCheckResult:
    found_numbers: list[str] = field(default_factory=list)
    valid_format: bool = False
    matched_claimed: bool = False
    claimed_permit_number: str | None = None
    best_match: str | None = None
    similarity: float = 0.0
    raw_text: str = ""

    @property
    def status(self) -> str:
        if not self.found_numbers:
            return "fail"  # no permit-shaped string anywhere in the bundle
        if not self.valid_format:
            return "fail"
        if self.claimed_permit_number and not self.matched_claimed:
            return "review"  # present + valid format, but doesn't match the ad
        return "pass"


def _normalize(s: str) -> str:
    return re.sub(r"[\s\-]", "", s).upper()


def find_permit_numbers(text: str) -> list[str]:
    """Pull every permit-shaped substring out of OCR'd text."""
    return [m.group(0) for m in PERMIT_REGEX.finditer(text)]


def check_permit(
    image_path: str,
    claimed_permit_number: str | None = None,
    backend: OCRBackend | None = None,
) -> PermitCheckResult:
    """Run OCR on a listing image and validate the permit number found in it
    against what the listing claims.
    """
    backend = backend or get_backend("auto")
    raw_text = backend.extract_text(image_path)
    found = find_permit_numbers(raw_text)

    result = PermitCheckResult(
        found_numbers=found,
        claimed_permit_number=claimed_permit_number,
        raw_text=raw_text,
    )
    if not found:
        return result

    result.valid_format = True  # matched PERMIT_REGEX by construction

    if claimed_permit_number:
        claimed_norm = _normalize(claimed_permit_number)
        best_ratio, best = 0.0, None
        for candidate in found:
            ratio = SequenceMatcher(None, claimed_norm, _normalize(candidate)).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, candidate
        result.best_match = best
        result.similarity = best_ratio
        result.matched_claimed = best_ratio >= 0.9  # near-exact, tolerates OCR noise
    else:
        result.matched_claimed = True  # nothing to cross-check against

    return result