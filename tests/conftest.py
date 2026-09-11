import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def tmp_image(tmp_path):
    """A plain solid-color image with no permit text and no watermark."""
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (200, 200), (200, 200, 200)).save(path)
    return str(path)


@pytest.fixture
def tmp_image_with_text(tmp_path):
    def _make(text: str, name: str = "with_text.jpg") -> str:
        path = tmp_path / name
        img = Image.new("RGB", (400, 100), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 30), text, fill=(0, 0, 0))
        img.save(path)
        return str(path)

    return _make
