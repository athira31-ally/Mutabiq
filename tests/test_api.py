import json

from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ListingMetadata
from app.services.duplicates import DuplicateIndex
from app.services.pipeline import CompliancePipeline
import app.api.routes as routes


def test_api_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_inspect(listing: ListingMetadata, interior_bgr, monkeypatch) -> None:
    from PIL import Image
    import io

    pipe = CompliancePipeline(
        ocr_fn=lambda img: "Permit 7113984521 Horizon Gate Real Estate",
        detect_fn=lambda img: [],
        duplicate_index=DuplicateIndex(path=None),
    )
    monkeypatch.setattr(routes, "_pipeline", pipe)
    buf = io.BytesIO()
    Image.fromarray(interior_bgr[:, :, ::-1]).save(buf, format="JPEG")
    client = TestClient(app)
    response = client.post(
        "/v1/listings/inspect",
        data={"listing": listing.model_dump_json()},
        files={"images": ("living.jpg", buf.getvalue(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["listing_id"] == listing.listing_id
    assert body["compliant"] is True
    assert "checks" in body
    json.dumps(body)
