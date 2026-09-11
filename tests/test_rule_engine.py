from src.dup_hash import DuplicateMatch
from src.ocr_permit import PermitCheckResult
from src.rule_engine import Status, evaluate
from src.watermark_detector import Detection


def _permit(found=True, valid=True, matched=True, claimed="1239982634"):
    return PermitCheckResult(
        found_numbers=["1239982634"] if found else [],
        valid_format=valid,
        matched_claimed=matched,
        claimed_permit_number=claimed,
    )


def test_all_clean_passes():
    report = evaluate("L1", _permit(), [], [])
    assert report.status == Status.PASS
    assert report.violations == []


def test_missing_permit_is_hard_fail():
    report = evaluate("L1", _permit(found=False), [], [])
    assert report.status == Status.FAIL
    assert any(v.code == "PERMIT_MISSING" for v in report.violations)


def test_permit_mismatch_is_review():
    report = evaluate("L1", _permit(matched=False), [], [])
    assert report.status == Status.REVIEW
    assert any(v.code == "PERMIT_MISMATCH" for v in report.violations)


def test_watermark_detected_is_review():
    det = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
    report = evaluate("L1", _permit(), [det], [])
    assert report.status == Status.REVIEW
    assert any(v.code == "WATERMARK_DETECTED" for v in report.violations)


def test_cross_listing_duplicate_is_review():
    match = DuplicateMatch(image_id="x", listing_id="OTHER", agent_id="agent-2", distance=2)
    report = evaluate("L1", _permit(), [], [match])
    assert report.status == Status.REVIEW
    assert any(v.code == "DUPLICATE_PHOTO" for v in report.violations)


def test_fail_outranks_review():
    det = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.9)
    report = evaluate("L1", _permit(found=False), [det], [])
    assert report.status == Status.FAIL