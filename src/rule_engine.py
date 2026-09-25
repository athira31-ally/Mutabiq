"""Combines the three checks (permit, watermark, duplicate photo) into one
structured compliance report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .dup_hash import DuplicateMatch
from .ocr_permit import PermitCheckResult
from .qr_permit import PermitQR
from .watermark_detector import Detection


class Status(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


@dataclass
class Violation:
    code: str
    severity: Status
    message: str


@dataclass
class ComplianceReport:
    listing_id: str
    status: Status
    violations: list[Violation] = field(default_factory=list)
    permit_check: PermitCheckResult | None = None
    watermark_detections: list[Detection] = field(default_factory=list)
    duplicate_matches: list[DuplicateMatch] = field(default_factory=list)
    permit_qrs: list[PermitQR] = field(default_factory=list)
    own_branding: list[dict] = field(default_factory=list)   # the listing agency's own logos (allowed)

    def to_dict(self) -> dict:
        return {
            "listing_id": self.listing_id,
            "status": self.status.value,
            "violations": [
                {"code": v.code, "severity": v.severity.value, "message": v.message}
                for v in self.violations
            ],
            "permit": {
                "found_numbers": self.permit_check.found_numbers if self.permit_check else [],
                "valid_format": self.permit_check.valid_format if self.permit_check else False,
                "matched_claimed": self.permit_check.matched_claimed if self.permit_check else False,
                "source": self.permit_source,
                "qr_codes": [q.to_dict() for q in self.permit_qrs if q.is_permit],
            },
            "watermarks_detected": len(self.watermark_detections),
            "own_branding": self.own_branding,
            "duplicate_matches": len(self.duplicate_matches),
        }


    @property
    def permit_source(self) -> str:
        """Where the permit was found: 'printed' (OCR), 'qr', both, or 'none'."""
        printed = bool(self.permit_check and self.permit_check.found_numbers)
        qr = any(q.is_permit for q in self.permit_qrs)
        return {(True, True): "printed+qr", (True, False): "printed", (False, True): "qr"}.get((printed, qr), "none")


def _worse(a: Status, b: Status) -> Status:
    order = {Status.PASS: 0, Status.REVIEW: 1, Status.FAIL: 2}
    return a if order[a] >= order[b] else b


def evaluate(
    listing_id: str,
    permit_check: PermitCheckResult,
    watermark_detections: list[Detection],
    duplicate_matches: list[DuplicateMatch],
    permit_qrs: list[PermitQR] | None = None,
    link_listing_ref: str | None = None,
    own_branding: list[dict] | None = None,
) -> ComplianceReport:
    """Rule table (see ARCHITECTURE.md §3.4):

    - No printed permit number AND no permit QR code  -> hard fail
    - Permit QR only (the portal style)               -> ok; if the QR carries a permit number,
                                                         it must match the claimed one (else review)
    - Permit found but malformed                      -> hard fail
    - Permit found + valid, but mismatches ad text     -> review
    - Permit QR belongs to a different listing than the pasted link -> review
    - Watermark detected (not the agency's own logo)   -> review
    - Duplicate photo matched to a different listing   -> review
    - All clean                                        -> pass
    """
    violations: list[Violation] = []
    overall = Status.PASS

    permit_qrs = permit_qrs or []
    qr_permits = [q for q in permit_qrs if q.is_permit]

    if not permit_check.found_numbers and not qr_permits:
        violations.append(
            Violation("PERMIT_MISSING", Status.FAIL, "No permit number or Trakheesi permit QR code found in the listing.")
        )
        overall = _worse(overall, Status.FAIL)
    elif not permit_check.found_numbers:
        qr_numbers = [q.permit_number for q in qr_permits if q.permit_number]
        claimed = permit_check.claimed_permit_number
        if claimed and qr_numbers and claimed.strip() not in qr_numbers:
            violations.append(
                Violation(
                    "PERMIT_MISMATCH",
                    Status.REVIEW,
                    f"Permit QR points to permit '{qr_numbers[0]}', but the ad claims '{claimed}'.",
                )
            )
            overall = _worse(overall, Status.REVIEW)
    elif not permit_check.valid_format:
        violations.append(
            Violation("PERMIT_MALFORMED", Status.FAIL, "Permit-shaped text found but doesn't match the expected format.")
        )
        overall = _worse(overall, Status.FAIL)
    elif permit_check.claimed_permit_number and not permit_check.matched_claimed:
        violations.append(
            Violation(
                "PERMIT_MISMATCH",
                Status.REVIEW,
                f"OCR'd permit '{permit_check.best_match}' doesn't match claimed "
                f"'{permit_check.claimed_permit_number}' (similarity {permit_check.similarity:.2f}).",
            )
        )
        overall = _worse(overall, Status.REVIEW)

    qr_refs = {q.listing_ref for q in qr_permits if q.listing_ref}
    if link_listing_ref and qr_refs and link_listing_ref not in qr_refs:
        violations.append(
            Violation(
                "PERMIT_QR_OTHER_LISTING",
                Status.REVIEW,
                f"The permit QR is for listing {sorted(qr_refs)[0]}, but this is listing {link_listing_ref} "
                "(a permit copied from another ad).",
            )
        )
        overall = _worse(overall, Status.REVIEW)

    if watermark_detections:
        violations.append(
            Violation(
                "WATERMARK_DETECTED",
                Status.REVIEW,
                f"{len(watermark_detections)} unauthorized watermark/logo region(s) detected.",
            )
        )
        overall = _worse(overall, Status.REVIEW)

    cross_listing = [m for m in duplicate_matches if m.listing_id != listing_id]
    if cross_listing:
        violations.append(
            Violation(
                "DUPLICATE_PHOTO",
                Status.REVIEW,
                f"Photo matches {len(cross_listing)} image(s) from other listing(s)/agent(s) "
                f"(closest: listing {cross_listing[0].listing_id}, agent {cross_listing[0].agent_id}).",
            )
        )
        overall = _worse(overall, Status.REVIEW)

    return ComplianceReport(
        listing_id=listing_id,
        status=overall,
        violations=violations,
        permit_check=permit_check,
        watermark_detections=watermark_detections,
        duplicate_matches=duplicate_matches,
        permit_qrs=permit_qrs,
        own_branding=own_branding or [],
    )
