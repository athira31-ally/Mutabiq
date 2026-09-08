"""Perceptual-hash duplicate / stolen-photo detection."""

from __future__ import annotations

import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from app.models.schemas import CheckResult


def phash_hex(bgr: np.ndarray) -> str:
    rgb = Image.fromarray(bgr[:, :, ::-1])
    return str(imagehash.phash(rgb))


def hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


class DuplicateIndex:
    def __init__(self, path: Path | None = None, threshold: int = 8) -> None:
        self.path = path
        self.threshold = threshold
        self._hashes: dict[str, str] = {}  # phash -> listing_id
        if path and path.exists():
            self._hashes = json.loads(path.read_text())

    def nearest(self, digest: str, exclude_listing: str | None = None) -> tuple[str | None, int]:
        best_id: str | None = None
        best_dist = 64
        for stored, listing_id in self._hashes.items():
            if exclude_listing and listing_id == exclude_listing:
                continue
            dist = hamming(digest, stored)
            if dist < best_dist:
                best_dist = dist
                best_id = listing_id
        return best_id, best_dist

    def add(self, digest: str, listing_id: str) -> None:
        self._hashes[digest] = listing_id

    def persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._hashes, indent=2))

    def check(self, digest: str, listing_id: str) -> CheckResult:
        other, dist = self.nearest(digest, exclude_listing=listing_id)
        duplicate = other is not None and dist <= self.threshold
        return CheckResult(
            name="duplicate_photo",
            passed=not duplicate,
            score=0.0 if duplicate else 1.0,
            detail=(
                f"Near-duplicate of listing {other} (Hamming {dist})."
                if duplicate
                else f"No near-duplicate in index (nearest Hamming {dist})."
            ),
            evidence={"phash": digest, "nearest_listing": other, "hamming": dist},
        )
