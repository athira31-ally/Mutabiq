import numpy as np

from training.synthesize import GENERIC_MARKS, overlay_logo, render_generic_logo, write_yolo_label


def test_overlay_produces_yolo_label_in_bounds(interior_bgr: np.ndarray, tmp_path) -> None:
    out, box = overlay_logo(interior_bgr, seed=7)
    cx, cy, bw, bh = box
    assert out.shape == interior_bgr.shape
    assert 0 < cx < 1 and 0 < cy < 1
    assert 0 < bw < 1 and 0 < bh < 1
    label = tmp_path / "0.txt"
    write_yolo_label(label, box)
    parts = label.read_text().split()
    assert parts[0] == "0"
    assert len(parts) == 5


def test_synthetic_logo_is_not_trademarked_wordmark() -> None:
    banned = {"emaar", "bayut", "dubizzle", "propertyfinder", "damac", "nakheel"}
    logo = render_generic_logo(seed=3)
    assert logo.mode == "RGBA"
    assert banned.isdisjoint({m.lower() for m in GENERIC_MARKS})
