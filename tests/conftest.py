from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from PIL import Image

from app.models.schemas import ListingMetadata
from scripts.create_dummy_onnx import write_dummy_yolo_onnx


@pytest.fixture
def listing() -> ListingMetadata:
    return ListingMetadata(
        listing_id="dxb-marina-2br-8841",
        permit_number="7113984521",
        broker_name="Horizon Gate Real Estate",
        developer_name="Emaar Properties",
        community="Dubai Marina",
        permit_issued=date(2026, 1, 12),
        permit_expires=date(2027, 1, 11),
        asking_price_aed=2_150_000,
        contact_phone="+971501234567",
    )


@pytest.fixture
def interior_bgr() -> np.ndarray:
    img = Image.new("RGB", (640, 480), (210, 200, 190))
    return np.array(img)[:, :, ::-1].copy()


@pytest.fixture(scope="session")
def dummy_onnx(tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("onnx") / "dummy.onnx"
    write_dummy_yolo_onnx(path, height=320, width=320)
    return path
