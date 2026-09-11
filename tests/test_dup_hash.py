import random

from PIL import Image

from src.dup_hash import DuplicatePhotoIndex


def _make_image(path, color):
    Image.new("RGB", (256, 256), color).save(path)


def _make_textured_image(path, seed):
    # Solid-color images are a degenerate case for perceptual hashing (almost
    # no frequency content, so two flat colors can hash identically) — use
    # actual texture here so the hash is meaningfully discriminative, the
    # same as it would be for real listing photos.
    rng = random.Random(seed)
    img = Image.new("RGB", (256, 256))
    pixels = img.load()
    for x in range(256):
        for y in range(256):
            pixels[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    img.save(path)


def test_no_match_for_unique_image(tmp_path):
    idx = DuplicatePhotoIndex(":memory:")
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _make_textured_image(img_a, seed=1)
    _make_textured_image(img_b, seed=2)

    idx.add("a0", str(img_a), listing_id="listing-1", agent_id="agent-1")
    matches = idx.find_matches(str(img_b))
    assert matches == []


def test_finds_cross_listing_duplicate(tmp_path):
    idx = DuplicatePhotoIndex(":memory:")
    original = tmp_path / "orig.jpg"
    reused = tmp_path / "reused.jpg"
    _make_image(original, (100, 150, 200))
    reused.write_bytes(original.read_bytes())  # byte-identical "reused" photo

    idx.add("orig0", str(original), listing_id="listing-1", agent_id="agent-A")
    matches = idx.find_matches(str(reused), exclude_listing_id="listing-2")

    assert len(matches) == 1
    assert matches[0].listing_id == "listing-1"
    assert matches[0].agent_id == "agent-A"
    assert matches[0].distance == 0


def test_excludes_same_listing():
    idx = DuplicatePhotoIndex(":memory:")
    # same listing re-checking its own already-registered photo shouldn't self-match
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path

        p = Path(d) / "img.jpg"
        _make_image(p, (50, 60, 70))
        idx.add("img0", str(p), listing_id="listing-1", agent_id="agent-1")
        matches = idx.find_matches(str(p), exclude_listing_id="listing-1")
        assert matches == []
