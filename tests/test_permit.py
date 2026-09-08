from datetime import date

from app.models.schemas import ListingMetadata
from app.services.permit import is_permit_shape, validate_permit_metadata


def test_valid_numeric_permit(listing: ListingMetadata) -> None:
    result = validate_permit_metadata(listing, today=date(2026, 6, 1))
    assert result.passed is True
    assert is_permit_shape(listing.permit_number)


def test_invalid_short_permit(listing: ListingMetadata) -> None:
    listing.permit_number = "12345"
    result = validate_permit_metadata(listing)
    assert result.passed is False


def test_expired_permit(listing: ListingMetadata) -> None:
    listing.permit_expires = date(2025, 12, 1)
    result = validate_permit_metadata(listing, today=date(2026, 6, 1))
    assert result.passed is False
    assert "expired" in result.detail.lower()
