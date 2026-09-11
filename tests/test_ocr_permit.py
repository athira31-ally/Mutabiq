from src.ocr_permit import MockOCR, check_permit, find_permit_numbers


def test_find_permit_numbers_extracts_shaped_strings():
    text = "Beautiful 2BR apartment. Permit #: 1239982634 Contact us today."
    found = find_permit_numbers(text)
    assert "1239982634" in found


def test_find_permit_numbers_empty_when_none_present():
    assert find_permit_numbers("Lovely marina view apartment for sale") == []


def test_check_permit_fail_when_no_permit_found(tmp_image):
    result = check_permit(tmp_image, backend=MockOCR("no permit info here"))
    assert result.status == "fail"
    assert result.found_numbers == []


def test_check_permit_pass_when_no_claim_to_cross_check(tmp_image):
    result = check_permit(tmp_image, backend=MockOCR("Permit # 1239982634 valid"))
    assert result.status == "pass"
    assert result.valid_format is True


def test_check_permit_review_on_mismatch(tmp_image):
    result = check_permit(
        tmp_image,
        claimed_permit_number="7169578165",
        backend=MockOCR("Permit # 1239982634 on file"),
    )
    assert result.status == "review"
    assert result.matched_claimed is False


def test_check_permit_pass_on_match(tmp_image):
    result = check_permit(
        tmp_image,
        claimed_permit_number="1239982634",
        backend=MockOCR("Permit # 1239982634 on file"),
    )
    assert result.status == "pass"
    assert result.matched_claimed is True