"""Download real, freely-licensed property/interior photos to use as the
*base* images for the watermark training set — replacing the procedurally
generated rooms in `synthetic_watermark_data.py` with actual photographs.
The logo overlaid on top stays synthetic either way (see that module's
docstring for why: using a real competitor's logo raises trademark
questions a generated wordmark doesn't).

Uses the Pexels API — free, no cost, no card required, generous limits
(200 requests/hour on the free tier, comfortably enough for this). Photos
come back under the Pexels License: free for commercial and personal use,
no attribution required, but redistributing an *unaltered* photo isn't
allowed — which is exactly why these are only ever used as a base layer
that gets a logo composited onto it for internal model training, never
published or served as-is.

Get a free key in about 30 seconds, no credit card: https://www.pexels.com/api/new/

Usage:
    export PEXELS_API_KEY=your_key_here
    python -m src.fetch_stock_photos --out data/raw/stock_photos --per-query 30

Needs a real internet connection — this only works run from your own
machine, not from a sandboxed build environment with restricted network
access (which is why it's written but wasn't run as part of building this
project; see README.md, "Known limitations").
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"

# Deliberately UAE/apartment-flavored search terms — the point is base photos
# that look like the kind of listing photo this tool is meant to check, not
# generic stock photography.
DEFAULT_QUERIES = [
    "modern apartment interior",
    "luxury living room",
    "empty apartment interior",
    "furnished apartment bedroom",
    "modern kitchen interior",
    "apartment balcony city view",
    "minimalist living room",
    "dubai apartment interior",
]


def fetch_query(
    query: str, per_query: int, api_key: str, out_dir: Path, seen_ids: set[int]
) -> int:
    """Download up to `per_query` new photos matching one search term.
    Skips any photo id already in `seen_ids` (shared across queries so the
    same photo doesn't get pulled twice if two search terms overlap).
    """
    import requests

    headers = {"Authorization": api_key}
    downloaded = 0
    page = 1
    per_page = min(per_query, 80)  # 80 is Pexels' own per-page ceiling

    while downloaded < per_query:
        resp = requests.get(
            PEXELS_SEARCH_URL,
            headers=headers,
            params={
                "query": query,
                "per_page": per_page,
                "page": page,
                "orientation": "landscape",
            },
            timeout=20,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        if not photos:
            break  # ran out of results for this query before hitting per_query

        for photo in photos:
            if downloaded >= per_query:
                break
            pid = photo["id"]
            if pid in seen_ids:
                continue
            seen_ids.add(pid)

            img_resp = requests.get(photo["src"]["large"], timeout=20)
            img_resp.raise_for_status()

            safe_query = query.replace(" ", "_")
            path = out_dir / f"{safe_query}_{pid}.jpg"
            path.write_bytes(img_resp.content)
            downloaded += 1
            time.sleep(0.15)  # stay well under the rate limit, no need to hurry

        page += 1

    return downloaded


def fetch_all(
    out_dir: str | Path,
    queries: list[str],
    per_query: int,
    api_key: str | None = None,
) -> int:
    api_key = api_key or os.environ.get("PEXELS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Pexels API key found. Get a free one at "
            "https://www.pexels.com/api/new/ then either pass --api-key "
            "or `export PEXELS_API_KEY=...` first."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()
    total = 0
    for query in queries:
        n = fetch_query(query, per_query, api_key, out_dir, seen_ids)
        print(f"  '{query}': {n} photos")
        total += n
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/stock_photos")
    parser.add_argument("--per-query", type=int, default=25)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    total = fetch_all(args.out, args.queries, args.per_query, api_key=args.api_key)
    print(f"\nDone — {total} real property/interior photos saved to {args.out}")