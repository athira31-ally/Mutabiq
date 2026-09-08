"""Optional demo: pull a Pexels interior and run the compliance pipeline."""

from __future__ import annotations

import argparse
import os
from datetime import date

import cv2
import numpy as np

from app.models.schemas import ListingMetadata
from app.services.pipeline import CompliancePipeline
from training.fetch_pexels import fetch_photos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("PEXELS_API_KEY", ""))
    parser.add_argument("--dest", default="data/pexels")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("Set PEXELS_API_KEY or pass --api-key")
    from pathlib import Path

    photos = fetch_photos(args.api_key, Path(args.dest), per_query=1)
    bgr = cv2.imread(str(photos[0]))
    if bgr is None:
        raise SystemExit("Failed to read Pexels photo")
    listing = ListingMetadata(
        listing_id="pexels-demo",
        permit_number="7113984521",
        broker_name="Horizon Gate Real Estate",
        permit_expires=date(2027, 1, 11),
    )
    # Demo creative text is injected because stock interiors have no Trakheesi overlay.
    report = CompliancePipeline(
        ocr_fn=lambda img: "Permit 7113984521 Horizon Gate Real Estate"
    ).inspect(listing, [(photos[0].name, bgr)])
    print(report.model_dump_json(indent=2))
    _ = np.mean(bgr)


if __name__ == "__main__":
    main()
