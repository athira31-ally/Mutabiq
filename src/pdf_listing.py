"""Read a whole listing page saved as a PDF (in the browser: Cmd+P -> Save as PDF).

One upload gives us everything a checker needs: the page text (Regulatory Information: permit, agency,
RERA / BRN numbers), the permit QR code, and the listing photos.

- **Text + QR codes:** Azure AI Document Intelligence (`prebuilt-layout` with the barcode add-on) when
  AZURE_DOCINTEL_ENDPOINT and AZURE_DOCINTEL_KEY are set. Otherwise local: the PDF's own text layer
  (pypdfium2), Tesseract for scanned pages, and zxing-cpp on the rendered pages for QR codes.
- **Photos:** always local - the images embedded in the PDF, skipping small ones (icons, logos), QR codes, and
  everything after a "Recommended for you"-style heading (other listings' photos).
  Only real photos go to the watermark and reused-photo checks; running those on whole rendered web pages
  flags buttons and headings as "logos".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .qr_permit import PermitQR, parse_qr_text

MAX_PAGES = 12
MAX_PHOTOS = 15
MIN_PHOTO_SIDE = 300      # px; smaller embedded images are icons, logos or the QR itself
RENDER_SCALE = 2.0        # 144 dpi: enough pixels per QR module for zxing


@dataclass
class PageExtract:
    text: str = ""
    qrs: list[PermitQR] = field(default_factory=list)
    photo_paths: list[str] = field(default_factory=list)
    pages: int = 0
    reader: str = "local"      # "azure-document-intelligence" or "local"


_NEXT_LABEL = r"(?=\s+(?:Zone\s+Name|Registered\s+Agency|RERA|ORN|BRN|Trakheesi|Permit|DED|$))"
FACT_PATTERNS = {  # applied to whitespace-collapsed text: a saved PDF wraps labels and values over lines
    "agency": r"Registered\s+Agency\s*:?\s*(.{3,80}?)" + _NEXT_LABEL,
    "rera": r"\b(?:RERA|ORN)\s*(?:No\.?|Number)?\s*:?\s*(\d{3,8})\b",
    "brn": r"\bBRN\s*:?\s*(\d{3,8})\b",
    "zone": r"Zone\s+Name\s*:?\s*(.{3,60}?)" + _NEXT_LABEL,
}


def regulatory_facts(text: str) -> dict:
    """The 'Regulatory Information' box as a dict (agency, RERA / ORN, BRN, zone) - whatever is present."""
    flat = " ".join(text.split())
    out = {}
    for key, pattern in FACT_PATTERNS.items():
        m = re.search(pattern, flat, re.I)
        if m:
            out[key] = m.group(1).strip()
    return out


def _docintel_configured() -> bool:
    return bool(os.environ.get("AZURE_DOCINTEL_ENDPOINT") and os.environ.get("AZURE_DOCINTEL_KEY"))


def _read_with_docintel(pdf_bytes: bytes) -> tuple[str, list[str]]:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(os.environ["AZURE_DOCINTEL_ENDPOINT"],
                                        AzureKeyCredential(os.environ["AZURE_DOCINTEL_KEY"]))
    poller = client.begin_analyze_document(
        "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=pdf_bytes),
        features=[DocumentAnalysisFeature.BARCODES], pages=f"1-{MAX_PAGES}")
    result = poller.result()
    codes = [b.value for p in (result.pages or []) for b in (p.barcodes or []) if b.value]
    return result.content or "", codes


def _read_locally(pdf) -> tuple[str, list[str]]:
    import zxingcpp

    texts, codes = [], []
    for i in range(min(len(pdf), MAX_PAGES)):
        page = pdf[i]
        text = page.get_textpage().get_text_range()
        img = page.render(scale=RENDER_SCALE).to_pil().convert("RGB")
        if len(text.strip()) < 20:   # scanned / image-only page: OCR it
            try:
                import pytesseract
                text = pytesseract.image_to_string(img)
            except Exception:  # pragma: no cover - tesseract missing
                pass
        texts.append(text)
        codes += [r.text for r in zxingcpp.read_barcodes(img) if r.text]
    return "\n".join(texts), codes


# Portals print other listings below the ad ("Recommended for you" on Bayut). Their photos are not part of
# this listing, and their agencies' logos would be flagged as watermarks on it - found on a real Bayut PDF.
OTHER_LISTINGS_HEADINGS = ("Recommended for you", "Similar properties", "Similar listings",
                           "You may also like", "Recently viewed", "More properties")


def _other_listings_start(pdf) -> tuple[int, float] | None:
    """(page index, y of the heading's top) where other listings begin, or None. PDF y grows upwards."""
    for i in range(min(len(pdf), MAX_PAGES)):
        textpage = pdf[i].get_textpage()
        hits = []
        for heading in OTHER_LISTINGS_HEADINGS:
            found = textpage.search(heading, match_case=False).get_next()
            if found:
                hits.append(textpage.get_charbox(found[0])[3])
        if hits:
            return i, max(hits)
    return None


def _extract_photos(pdf, out_dir: Path) -> list[str]:
    import pypdfium2.raw as pdfium_c
    import zxingcpp

    paths = []
    cut = _other_listings_start(pdf)
    for i in range(min(len(pdf), MAX_PAGES)):
        if cut and i > cut[0]:
            break
        for obj in pdf[i].get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
            if len(paths) >= MAX_PHOTOS:
                return paths
            if cut and i == cut[0] and obj.get_bounds()[1] < cut[1]:
                continue                        # at or below the "other listings" heading
            try:
                img = obj.get_bitmap(render=False).to_pil().convert("RGB")
            except Exception:
                continue
            if min(img.size) < MIN_PHOTO_SIDE:
                continue
            if zxingcpp.read_barcodes(img):      # the permit QR itself (can be a large, sharp image)
                continue
            path = out_dir / f"pdf_photo_{len(paths)}.jpg"
            img.save(path, quality=92)
            paths.append(str(path))
    return paths


def extract_pdf(pdf_path: str | Path, out_dir: str | Path) -> PageExtract:
    import pypdfium2 as pdfium

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(pdf_path).read_bytes()
    pdf = pdfium.PdfDocument(data)
    try:
        reader = "local"
        if _docintel_configured():
            try:
                text, codes = _read_with_docintel(data)
                reader = "azure-document-intelligence"
            except Exception:  # service down / quota: the local reader still gives an answer
                text, codes = _read_locally(pdf)
        else:
            text, codes = _read_locally(pdf)
        if reader != "local" and not codes:
            codes = _read_locally(pdf)[1]      # second opinion on the QR if the service found none
        photos = _extract_photos(pdf, out)
        pages = len(pdf)
    finally:
        pdf.close()
    qrs, seen = [], set()
    for c in codes:
        if c not in seen:
            seen.add(c)
            qrs.append(parse_qr_text(c))
    return PageExtract(text=text, qrs=qrs, photo_paths=photos, pages=pages, reader=reader)
