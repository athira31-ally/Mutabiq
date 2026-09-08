from app.models.schemas import ListingMetadata
from app.services.fuzzy import reconcile
from app.services.ocr import extract_permit_candidates

MARINA_OCR = (
    "AED 2,150,000  +971 50 123 4567  Permit  7113984521  Horizon Gate Real Estate"
)


def test_permit_extracted_from_noisy_ocr() -> None:
    assert extract_permit_candidates(MARINA_OCR) == ["7113984521"]


def test_phone_not_selected_as_permit() -> None:
    candidates = extract_permit_candidates("+971 50 123 4567 only")
    assert candidates == []
    assert not any(c.startswith("971") for c in candidates)


def test_price_not_selected_as_permit() -> None:
    assert extract_permit_candidates("AED 2,150,000") == []


def test_zero_oh_substitution(listing: ListingMetadata) -> None:
    listing.permit_number = "7113984501"
    result = reconcile("Permit 71139845O1", listing)
    assert result.passed is True


def test_broker_name_fuzzy(listing: ListingMetadata) -> None:
    result = reconcile(MARINA_OCR, listing)
    assert result.evidence["broker_score"] >= 80
    assert result.passed is True


def test_unrelated_ocr_fails(listing: ListingMetadata) -> None:
    result = reconcile("Sea view balcony gym pool", listing)
    assert result.passed is False


def test_arabic_label_extracts_permit() -> None:
    text = "??? ??????? 7113984521"
    assert "7113984521" in extract_permit_candidates(text)
