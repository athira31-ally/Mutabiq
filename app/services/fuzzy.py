"""Fuzzy reconciliation of OCR against listing metadata.

Original design (wrong): treat the entire OCR blob as a single string and
require exact equality with ``listing.permit_number``.

A real Dubai Marina portal listing broke that: the creative contained the
permit *and* an AED price *and* a +971 mobile. Exact match always failed,
and the first long digit run was often the phone number.

Redesign: classify tokens, score permit candidates separately from broker /
developer names, and require a field-aware permit hit.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from app.models.schemas import CheckResult, ListingMetadata
from app.services.ocr import extract_permit_candidates, fold_confusables, normalize_ocr_text
from app.services.permit import normalize_permit

PERMIT_MATCH_CUTOFF = 88
NAME_MATCH_CUTOFF = 80


def _best_permit_score(candidates: list[str], declared: str) -> tuple[float, str | None]:
    target = normalize_permit(fold_confusables(declared))
    best_score = 0.0
    best_token: str | None = None
    for token in candidates:
        score = fuzz.ratio(token, target)
        # Prefix/suffix OCR truncation of long DLD refs
        partial = fuzz.partial_ratio(token, target)
        score = max(score, partial * 0.98)
        if score > best_score:
            best_score = float(score)
            best_token = token
    return best_score, best_token


def _name_score(ocr_text: str, name: str | None) -> float:
    if not name:
        return 100.0
    haystack = normalize_ocr_text(ocr_text).lower()
    return float(fuzz.token_set_ratio(name.lower(), haystack))


def reconcile(ocr_text: str, listing: ListingMetadata) -> CheckResult:
    candidates = extract_permit_candidates(ocr_text)
    permit_score, matched = _best_permit_score(candidates, listing.permit_number)
    broker_score = _name_score(ocr_text, listing.broker_name)
    developer_score = _name_score(ocr_text, listing.developer_name)

    # Weighted: permit identity dominates; names are supporting evidence only.
    composite = (0.75 * permit_score) + (0.15 * broker_score) + (0.10 * developer_score)
    permit_ok = permit_score >= PERMIT_MATCH_CUTOFF
    passed = permit_ok

    if not candidates:
        detail = (
            "No Trakheesi-shaped number found in on-image text. "
            "Price and phone tokens are excluded from the permit pool."
        )
    elif not permit_ok:
        detail = (
            f"On-image permit candidates {candidates} do not match listing "
            f"permit {listing.permit_number}."
        )
    else:
        detail = f"OCR permit {matched} reconciles with listing metadata."

    return CheckResult(
        name="ocr_metadata_reconcile",
        passed=passed,
        score=round(min(composite, 100.0) / 100.0, 4),
        detail=detail,
        evidence={
            "candidates": candidates,
            "matched_permit": matched,
            "permit_score": permit_score,
            "broker_score": broker_score,
            "developer_score": developer_score,
            "cutoffs": {"permit": PERMIT_MATCH_CUTOFF, "name": NAME_MATCH_CUTOFF},
        },
    )
