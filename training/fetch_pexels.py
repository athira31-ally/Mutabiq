"""Download real interior / apartment photos from Pexels for detector training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import httpx

QUERIES = (
    "modern apartment interior",
    "living room apartment",
    "bedroom interior apartment",
    "kitchen apartment interior",
    "dubai apartment interior",
)


def fetch_photos(api_key: str, dest: Path, per_query: int = 15) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": api_key}
    saved: list[Path] = []
    with httpx.Client(timeout=60.0, headers=headers, follow_redirects=True) as client:
        for query in QUERIES:
            response = client.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": per_query, "orientation": "landscape"},
            )
            response.raise_for_status()
            for photo in response.json().get("photos", []):
                url = photo["src"].get("large2x") or photo["src"]["large"]
                path = dest / f"pexels_{photo['id']}.jpg"
                if path.exists():
                    saved.append(path)
                    continue
                img = client.get(url)
                img.raise_for_status()
                path.write_bytes(img.content)
                saved.append(path)
                time.sleep(0.15)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Pexels interior photos")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--dest", type=Path, default=Path("data/pexels"))
    parser.add_argument("--per-query", type=int, default=15)
    args = parser.parse_args()
    paths = fetch_photos(args.api_key, args.dest, args.per_query)
    print(f"Saved {len(paths)} photos to {args.dest}")


if __name__ == "__main__":
    main()
