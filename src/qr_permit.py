"""Trakheesi permit QR codes.

Dubai listings no longer have to print the permit number: DLD requires a permit QR code, and portals
show it in the listing's "Regulatory Information" box. Testing on a real Bayut listing showed the QR
holds a validation link, not a bare number:

    https://www.bayut.com/api/listing/<listing id>/permitValidation/<signature>

The last part is a digital signature (base64url ECDSA; it starts with "MEUCI..."). So the check here is
"is there a permit QR, and does it point at this listing / permit?". We can't verify the signature itself,
because that needs the portal's or DLD's public key; following the link is the authoritative check.

Decoding uses zxing-cpp (self-contained wheels for macOS and Linux, no system library needed). If it
isn't installed the check quietly returns no QR codes, and the printed-number OCR check still runs.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from PIL import Image

PERMIT_HINTS = ("permitvalidation", "trakheesi", "dubailand.gov.ae", "madmoun", "madhmoun")
LISTING_RE = re.compile(r"/listing/([A-Za-z0-9-]+)/", re.I)
PERMIT_PARAM_RE = re.compile(r"(?:permit(?:_?(?:number|no))?|trakheesi)[=/](\d{8,12})\b", re.I)
SIGNATURE_RE = re.compile(r"permitValidation/([A-Za-z0-9_\-=]{40,})", re.I)


@dataclass
class PermitQR:
    text: str
    is_permit: bool                    # looks like a permit / validation QR (not a random website link)
    listing_ref: str | None = None     # portal listing id in the link, if any
    permit_number: str | None = None   # permit number in the link, if the link carries one
    signed: bool = False               # link carries a validation signature (present, not verified)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_qr_text(text: str) -> PermitQR:
    low = text.lower()
    listing = LISTING_RE.search(text)
    permit = PERMIT_PARAM_RE.search(text)
    return PermitQR(
        text=text,
        is_permit=any(h in low for h in PERMIT_HINTS),
        listing_ref=listing.group(1) if listing else None,
        permit_number=permit.group(1) if permit else None,
        signed=bool(SIGNATURE_RE.search(text)),
    )


def _decode(img: Image.Image) -> list[str]:
    import zxingcpp
    return [r.text for r in zxingcpp.read_barcodes(img) if r.text]


def read_permit_qrs(image_path: str) -> list[PermitQR]:
    """Every QR code in the image, parsed. The QR needs roughly 2+ pixels per module: a normal (retina)
    screenshot of a portal's permit box decodes; the same screenshot downscaled to half size does not."""
    try:
        import zxingcpp  # noqa: F401
    except ImportError:  # pragma: no cover
        return []
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        texts = _decode(img)
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(parse_qr_text(t))
    return out
