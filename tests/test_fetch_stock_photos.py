"""Tests the Pexels download logic against a mocked HTTP layer — no real
network call, no real API key needed to run these. What actually hits
pexels.com only happens when you run the script for real (see
DEPLOY_AZURE.md-style honesty: this sandbox can't reach the open internet,
so this is the closest thing to a real test of that code available here).
"""

from unittest.mock import MagicMock, patch

from src.fetch_stock_photos import fetch_all


def _fake_search_response(photos):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"photos": photos}
    return resp


def _fake_image_response(content: bytes = b"fake-jpeg-bytes"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = content
    return resp


def test_fetch_all_downloads_and_dedupes(tmp_path):
    photos_page = [
        {"id": 1, "src": {"large": "https://images.pexels.com/1.jpg"}},
        {"id": 2, "src": {"large": "https://images.pexels.com/2.jpg"}},
    ]

    with patch("requests.get") as mock_get:
        def side_effect(url, *args, **kwargs):
            if url.endswith("/search"):
                return _fake_search_response(photos_page)
            return _fake_image_response()

        mock_get.side_effect = side_effect

        total = fetch_all(
            out_dir=tmp_path,
            queries=["modern apartment interior"],
            per_query=2,
            api_key="fake-key-for-test",
        )

    assert total == 2
    saved = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert saved == ["modern_apartment_interior_1.jpg", "modern_apartment_interior_2.jpg"]


def test_fetch_all_raises_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    try:
        fetch_all(out_dir=tmp_path, queries=["x"], per_query=1, api_key=None)
        assert False, "expected RuntimeError for missing API key"
    except RuntimeError as e:
        assert "PEXELS_API_KEY" in str(e)