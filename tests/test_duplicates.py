import cv2
import numpy as np
from PIL import Image

from app.services.duplicates import DuplicateIndex, hamming, phash_hex


def _bgr(color: tuple[int, int, int], size: tuple[int, int] = (128, 96), *, stripe: int = 0) -> np.ndarray:
    img = Image.new("RGB", size, color)
    if stripe:
        px = img.load()
        for x in range(0, size[0], stripe):
            for y in range(size[1]):
                px[x, y] = (255 - color[0], color[1], 255 - color[2])
    return np.array(img)[:, :, ::-1].copy()


def test_identical_phash_duplicates() -> None:
    frame = _bgr((40, 80, 120))
    a, b = phash_hex(frame), phash_hex(frame.copy())
    assert hamming(a, b) == 0


def test_resized_near_duplicates() -> None:
    frame = _bgr((200, 40, 40), (256, 192))
    small = cv2.resize(frame, (128, 96))
    dist = hamming(phash_hex(frame), phash_hex(small))
    index = DuplicateIndex(threshold=8)
    index.add(phash_hex(frame), "listing-a")
    check = index.check(phash_hex(small), "listing-b")
    assert dist <= 8
    assert check.passed is False


def test_distinct_not_duplicates() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
    gradient = np.zeros((96, 128, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, 128, dtype=np.uint8)
    gradient[:, :, 1] = np.linspace(255, 0, 96, dtype=np.uint8)[:, None]
    gradient[:, :, 2] = 80
    index = DuplicateIndex(threshold=8)
    index.add(phash_hex(noisy), "listing-a")
    check = index.check(phash_hex(gradient), "listing-b")
    assert hamming(phash_hex(noisy), phash_hex(gradient)) > 8
    assert check.passed is True
