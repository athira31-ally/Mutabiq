"""Permit structure and expiry checks (DLD / RERA advertising permit)."""

from __future__ import annotations

import re
from datetime import date

from app.models.schemas import CheckResult, ListingMetadata

# Live Trakheesi numbers are opaque DLD references. Portal listings commonly
# show 8-12 digit strings; some e-permits use TRK-YYYY-######.
_NUMERIC = re.compile(r"^\d{8,12}$")
_TRK = re.compile(r"^TRK-20\d{2}-\d{6,10}$", re.IGNORECASE)


def normalize_permit(value: str) -> str:
    return re.sub(r"[\s\-_.]", "", value).upper()


def is_permit_shape(value: str) -> bool:
    token = normalize_permit(value)
    if _TRK.match(token) or (
        token.startswith("TRK") and token[3:].isdigit() and 8 <= len(token[3:]) <= 12
    ):
        return True
    return bool(_NUMERIC.match(token))


def validate_permit_metadata(listing: ListingMetadata, today: date | None = None) -> CheckResult:
    today = today or date.today()
    token = normalize_permit(listing.permit_number)
    if not is_permit_shape(listing.permit_number):
        return CheckResult(
            name="permit_format",
            passed=False,
            score=0.0,
            detail="Permit number is missing or not a plausible DLD/Trakheesi reference.",
            evidence={"permit_number": listing.permit_number},
        )
    if listing.permit_expires and listing.permit_expires < today:
        return CheckResult(
            name="permit_format",
            passed=False,
            score=0.2,
            detail="Trakheesi permit has expired.",
            evidence={"permit_expires": listing.permit_expires.isoformat(), "normalized": token},
        )
    return CheckResult(
        name="permit_format",
        passed=True,
        score=1.0,
        detail="Permit number has a valid shape and is not expired in listing metadata.",
        evidence={"normalized": token, "permit_class": listing.permit_class.value},
    )
