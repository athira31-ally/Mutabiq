"""Duplicate / reused-photo detection via perceptual hashing.

Every ingested image gets a perceptual hash (pHash). New images are checked
against the index by Hamming distance; a close match belonging to a
*different* listing/agent is the signal for a reused or stolen photo — the
same agent re-uploading their own shot isn't a violation.

v1 uses a flat in-memory/SQLite index (fine up to a few thousand images,
which covers a single-agency use case comfortably); FAISS or a proper vector
index is the scaling path once the corpus grows past that.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

DEFAULT_MATCH_THRESHOLD = 6  # Hamming distance; lower = stricter match


@dataclass
class DuplicateMatch:
    image_id: str
    listing_id: str
    agent_id: str
    distance: int

    @property
    def cross_listing(self) -> bool:
        return True  # populated by the caller relative to the query's own listing


def compute_hash(image_path: str) -> imagehash.ImageHash:
    with Image.open(image_path) as img:
        return imagehash.phash(img)


class DuplicatePhotoIndex:
    """SQLite-backed pHash index. `:memory:` for tests/ephemeral use, a file
    path for a persistent index across runs.
    """

    def __init__(self, db_path: str = ":memory:"):
        # FastAPI serves requests from several threads (event loop + threadpool), so the connection must be
        # usable across threads; the lock keeps concurrent reads/writes from interleaving.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                listing_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                phash TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, image_id: str, image_path: str, listing_id: str, agent_id: str) -> None:
        phash = str(compute_hash(image_path))
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO images (image_id, listing_id, agent_id, phash) VALUES (?, ?, ?, ?)",
                (image_id, listing_id, agent_id, phash),
            )
            self.conn.commit()

    def find_matches(
        self,
        image_path: str,
        exclude_listing_id: str | None = None,
        threshold: int = DEFAULT_MATCH_THRESHOLD,
    ) -> list[DuplicateMatch]:
        query_hash = compute_hash(image_path)
        matches = []
        with self.lock:
            rows = self.conn.execute("SELECT image_id, listing_id, agent_id, phash FROM images").fetchall()
        for image_id, listing_id, agent_id, phash_str in rows:
            if exclude_listing_id and listing_id == exclude_listing_id:
                continue
            distance = query_hash - imagehash.hex_to_hash(phash_str)
            if distance <= threshold:
                matches.append(
                    DuplicateMatch(
                        image_id=image_id,
                        listing_id=listing_id,
                        agent_id=agent_id,
                        distance=distance,
                    )
                )
        return sorted(matches, key=lambda m: m.distance)

    def close(self) -> None:
        self.conn.close()
