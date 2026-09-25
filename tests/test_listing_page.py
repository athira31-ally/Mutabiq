"""Checking a whole listing from its link and/or the page saved as a PDF. No network: the portal's
responses are faked, and the test PDF is built in the test (a photo + a permit QR, like a saved page)."""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import listing_link
from src.listing_link import fetch_listing, parse_listing_url
from src.pdf_listing import extract_pdf, regulatory_facts

zxingcpp = pytest.importorskip("zxingcpp")
pdfium = pytest.importorskip("pypdfium2")

MODEL = "models/watermark_yolov8n.onnx"
PHOTO = "data/raw/stock_photos/luxury_living_room_14613702.jpg"
QR_LINK = "https://portal.example.ae/api/listing/15605505/permitValidation/MEUCIDFQ" + "x" * 80


def _insert_image(pdf, page, img, x, y, width):
    obj = pdfium.PdfImage.new(pdf)
    obj.set_bitmap(pdfium.PdfBitmap.from_pil(img))
    obj.set_matrix(pdfium.PdfMatrix().scale(width, width * img.height / img.width).translate(x, y))
    page.insert_obj(obj)


@pytest.fixture
def listing_pdf(tmp_path):
    """A one-page 'saved listing': a big photo and a small permit QR."""
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(595, 842)
    photo = Image.open(PHOTO).convert("RGB")
    photo.thumbnail((900, 900))
    _insert_image(pdf, page, photo, 40, 420, 420)
    qr = Image.fromarray(np.array(zxingcpp.create_barcode(QR_LINK, zxingcpp.BarcodeFormat.QRCode).to_image(scale=6)))
    _insert_image(pdf, page, qr.convert("RGB"), 420, 120, 120)
    page.gen_content()
    path = tmp_path / "listing.pdf"
    pdf.save(path)
    return str(path)


# --- links -------------------------------------------------------------------------------------------

def test_listing_id_is_read_from_the_link():
    assert parse_listing_url("https://www.bayut.com/property/details-15605505.html").listing_ref == "15605505"
    pf = parse_listing_url("https://www.propertyfinder.ae/en/plp/rent/villa-for-rent-dubai-the-fields-14567890.html")
    assert pf.portal == "Property Finder" and pf.listing_ref == "14567890"


@pytest.mark.parametrize("url", ["https://evil.example.com/details-1.html", "file:///etc/passwd", "bayut.com"])
def test_only_portal_links_are_accepted(url):
    with pytest.raises(ValueError):
        parse_listing_url(url)


def test_blocked_portal_asks_for_the_pdf(monkeypatch, tmp_path):
    # What Bayut actually answered to a plain request from a home connection: 401, empty body.
    monkeypatch.setattr(listing_link, "_http_get", lambda url, limit: (200, b"") if url.endswith("robots.txt") else (401, b""))
    r = fetch_listing("https://www.bayut.com/property/details-15605505.html", tmp_path)
    assert not r.ok and "blocks automated requests" in r.reason and r.http_status == 401


def test_robots_disallow_is_respected(monkeypatch, tmp_path):
    calls = []

    def fake(url, limit):
        calls.append(url)
        return 200, b"User-agent: *\nDisallow: /property/\n"
    monkeypatch.setattr(listing_link, "_http_get", fake)
    r = fetch_listing("https://www.bayut.com/property/details-15605505.html", tmp_path)
    assert not r.ok and "robots.txt" in r.reason
    assert calls == ["https://www.bayut.com/robots.txt"]           # the page itself was never requested


def test_allowed_page_gives_photos_and_text(monkeypatch, tmp_path):
    page = b"""<html><head><meta property="og:image" content="https://images.bayut.com/1.jpg">
    <script type="application/ld+json">{"@type":"Residence","image":["https://images.bayut.com/2.jpg",
    "https://tracker.evil.com/x.jpg"]}</script></head><body><h3>Regulatory Information</h3>
    <div>Registered Agency</div><div>SEROVIA PROPERTIES L.L.C</div><div>BRN</div><div>84967</div></body></html>"""
    photo = open(PHOTO, "rb").read()

    def fake(url, limit):
        if url.endswith("robots.txt"):
            return 200, b"User-agent: *\nDisallow: /search/\n"
        return (200, page) if url.endswith(".html") else (200, photo)
    monkeypatch.setattr(listing_link, "_http_get", fake)
    r = fetch_listing("https://www.bayut.com/property/details-15605505.html", tmp_path)
    assert r.ok and len(r.image_paths) == 2                          # the off-portal image is not downloaded
    assert regulatory_facts(r.page_text) == {"agency": "SEROVIA PROPERTIES L.L.C", "brn": "84967"}


# --- PDF ---------------------------------------------------------------------------------------------

def test_pdf_gives_photos_and_the_permit_qr(listing_pdf, tmp_path):
    x = extract_pdf(listing_pdf, tmp_path / "out")
    assert len(x.photo_paths) == 1                                   # the photo; the small QR is not a photo
    assert [(q.is_permit, q.listing_ref) for q in x.qrs] == [(True, "15605505")]
    assert x.reader == "local"


def test_regulatory_facts_from_page_text():
    text = "Regulatory Information\nZone Name\nWadi Al Safa 3\nRegistered Agency\nSEROVIA PROPERTIES L.L.C\nRERA\n51885\nBRN\n84967"
    assert regulatory_facts(text) == {"agency": "SEROVIA PROPERTIES L.L.C", "rera": "51885",
                                      "brn": "84967", "zone": "Wadi Al Safa 3"}


# --- API ---------------------------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    if not os.path.exists(MODEL):
        pytest.skip("trained model not present")
    import src.api as api
    monkeypatch.setattr(api, "DUP_INDEX_PATH", str(tmp_path / "idx.sqlite"))
    monkeypatch.setattr(api, "_pipeline", None)
    return TestClient(api.app)


def test_link_plus_pdf_passes_and_rechecking_is_not_a_reuse(client, listing_pdf):
    for _ in range(2):
        r = client.post("/check-page", data={"listing_url": "https://www.bayut.com/property/details-15605505.html"},
                        files={"page_pdf": ("l.pdf", open(listing_pdf, "rb"), "application/pdf")}).json()
        assert r["status"] in ("pass", "review") and r["permit"]["source"] == "qr"
        assert not any(v["code"] in ("DUPLICATE_PHOTO", "PERMIT_MISSING", "PERMIT_QR_OTHER_LISTING")
                       for v in r["violations"])
        assert r["listing"]["listing_ref"] == "15605505" and r["listing"]["photos_checked"] == 1


def test_permit_qr_from_another_listing_is_flagged(client, listing_pdf):
    r = client.post("/check-page", data={"listing_url": "https://www.bayut.com/property/details-77777777.html"},
                    files={"page_pdf": ("l.pdf", open(listing_pdf, "rb"), "application/pdf")}).json()
    assert "PERMIT_QR_OTHER_LISTING" in [v["code"] for v in r["violations"]]


def test_link_only_on_a_blocking_portal_returns_needs_pdf(client, monkeypatch):
    import src.api as api
    monkeypatch.setattr(api, "fetch_listing", lambda url, out: listing_link.FetchResult(False, "Bayut blocks automated requests (HTTP 401)."))
    r = client.post("/check-page", data={"listing_url": "https://www.bayut.com/property/details-15605505.html"}).json()
    assert r["status"] == "needs_pdf" and "Save as PDF" in r["message"] and r["listing"]["listing_ref"] == "15605505"


def test_not_a_pdf_is_rejected(client):
    r = client.post("/check-page", files={"page_pdf": ("x.pdf", b"hello", "application/pdf")})
    assert r.status_code == 422


def test_regulatory_facts_when_the_pdf_wraps_labels_over_lines():
    # How Chrome's "Save as PDF" of a real Bayut listing lays the box out (line breaks inside labels).
    text = ("Regulatory\r\nInformation\r\nZone\r\nName\r\nWadi Al Safa 3\r\nRegistered\r\nAgency\r\nSEROVIA\r\n"
            "PROPERTIES L.L.C\r\nRERA 51885\r\nBRN 84967\r\nTrakheesi\r\nPermit\r\nRecommended for you")
    assert regulatory_facts(text) == {"agency": "SEROVIA PROPERTIES L.L.C", "rera": "51885",
                                      "brn": "84967", "zone": "Wadi Al Safa 3"}


def _add_text(pdf, page, text, x, y, size=14):
    import ctypes

    import pypdfium2.raw as raw
    obj = raw.FPDFPageObj_NewTextObj(pdf.raw, b"Helvetica", ctypes.c_float(size))
    buf = (ctypes.c_ushort * (len(text) + 1))(*[ord(ch) for ch in text], 0)
    raw.FPDFText_SetText(obj, ctypes.cast(buf, raw.FPDF_WIDESTRING))
    raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y)
    raw.FPDFPage_InsertObject(page.raw, obj)


def test_photos_of_other_recommended_listings_are_left_out(tmp_path):
    # A real Bayut PDF ends with "Recommended for you" thumbnails of other agencies' listings.
    pdf = pdfium.PdfDocument.new()
    photo = Image.open(PHOTO).convert("RGB")
    photo.thumbnail((900, 900))
    page = pdf.new_page(595, 842)
    _insert_image(pdf, page, photo, 40, 420, 420)            # this listing's photo
    _add_text(pdf, page, "Recommended for you", 40, 300)
    _insert_image(pdf, page, photo, 40, 60, 200)             # another listing's thumbnail, below the heading
    page.gen_content()
    path = tmp_path / "with_recs.pdf"
    pdf.save(path)
    assert len(extract_pdf(path, tmp_path / "out").photo_paths) == 1
