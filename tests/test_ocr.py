from app.services.ocr import extract_permit_candidates, fold_confusables, normalize_ocr_text


def test_normalize_collapses_spaces_and_dashes() -> None:
    text = "Permit   7113-984-521"
    assert "  " not in normalize_ocr_text(text)
    candidates = extract_permit_candidates(text)
    assert "7113984521" in candidates


def test_confusable_character_fold() -> None:
    assert fold_confusables("71139845O1") == "7113984501"
