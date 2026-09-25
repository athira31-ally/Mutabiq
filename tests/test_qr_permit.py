"""Permit QR codes: portals now show the Trakheesi permit as a QR code instead of a printed number.
The QR codes here are generated in the test (the link shape matches a real portal permit QR); no
real listing images are committed."""
import numpy as np
import pytest
from PIL import Image

from src.ocr_permit import MockOCR, check_permit, find_permit_numbers
from src.qr_permit import parse_qr_text, read_permit_qrs
from src.rule_engine import Status, evaluate

zxingcpp = pytest.importorskip("zxingcpp")

PORTAL_LINK = "https://portal.example.ae/api/listing/15605505/permitValidation/" + "MEUCIDFQ" + "x" * 80


def _photo_with_qr(tmp_path, text, name="listing.png", qr_px=180):
    qr = Image.fromarray(np.array(zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.QRCode).to_image(scale=6)))
    qr = qr.convert("RGB").resize((qr_px, qr_px), Image.NEAREST)
    img = Image.new("RGB", (800, 600), (214, 204, 188))
    img.paste(qr, (590, 390))
    path = tmp_path / name
    img.save(path)
    return str(path)


def test_parses_portal_permit_link():
    q = parse_qr_text(PORTAL_LINK)
    assert q.is_permit and q.signed and q.listing_ref == "15605505" and q.permit_number is None


def test_permit_number_in_link_is_extracted():
    q = parse_qr_text("https://trakheesi.dubailand.gov.ae/validate?permitNumber=7169578165")
    assert q.is_permit and q.permit_number == "7169578165"


def test_ordinary_website_qr_is_not_a_permit():
    assert not parse_qr_text("https://www.some-brokerage.ae/contact").is_permit


def test_reads_qr_from_listing_image(tmp_path):
    qrs = read_permit_qrs(_photo_with_qr(tmp_path, PORTAL_LINK))
    assert len(qrs) == 1 and qrs[0].is_permit and qrs[0].listing_ref == "15605505"


def test_qr_only_listing_passes_permit_check(tmp_path):
    path = _photo_with_qr(tmp_path, PORTAL_LINK)
    report = evaluate("L1", check_permit(path, backend=MockOCR("Private garden, 4 beds")), [], [],
                      permit_qrs=read_permit_qrs(path))
    assert report.status == Status.PASS
    assert report.to_dict()["permit"]["source"] == "qr"


def test_qr_permit_number_must_match_the_ad(tmp_path):
    path = _photo_with_qr(tmp_path, "https://trakheesi.dubailand.gov.ae/validate?permitNumber=7169578165")
    permit = check_permit(path, claimed_permit_number="1239982634", backend=MockOCR(""))
    report = evaluate("L1", permit, [], [], permit_qrs=read_permit_qrs(path))
    assert [v.code for v in report.violations] == ["PERMIT_MISMATCH"]


def test_no_printed_permit_and_no_qr_still_fails(tmp_image):
    report = evaluate("L1", check_permit(tmp_image, backend=MockOCR("")), [], [], permit_qrs=read_permit_qrs(tmp_image))
    assert report.status == Status.FAIL and report.violations[0].code == "PERMIT_MISSING"


def test_numbers_without_a_permit_label_are_ignored():
    # Real case: OCR of a Bayut screenshot picked up listing IDs from the browser's address bar.
    text = "bayut.com/property/details-15605505.html\nCall 0501234567\nTrakheesi Permit No. 7169578165"
    assert find_permit_numbers(text) == ["7169578165"]


def test_permit_label_on_the_line_above_counts():
    assert find_permit_numbers("Trakheesi Permit\n6045128830") == ["6045128830"]
