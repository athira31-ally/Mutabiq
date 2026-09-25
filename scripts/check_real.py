"""Run the compliance pipeline on your own real listing screenshots (kept out of git).

    python -m scripts.check_real real_tests
    python -m scripts.check_real real_tests --claimed 7169578165
    python -m scripts.check_real real_tests --one-listing      # every image in the folder = one listing

A real listing has several photos, so put each listing in its own subfolder:

    real_tests/
      bayut_15605505/        <- one listing: its photos and/or the page saved as PDF (Cmd+P -> Save as PDF)
      bayut_16382531/
      single_shot.png        <- loose images / PDFs are checked on their own
      expected.csv           <- optional: `bayut_15605505,pass` / `single_shot.png,fail`

A listing number in the folder name (bayut_15605505) is treated like the pasted link: the permit QR must
belong to that listing.

With expected.csv, extra columns show OK / WRONG and a score.
"""
from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path

from src.dup_hash import DuplicatePhotoIndex
from src.pdf_listing import extract_pdf, regulatory_facts
from src.pipeline import CompliancePipeline, ListingBundle

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".pdf"}


def _expected(folder: Path) -> dict[str, str]:
    f = folder / "expected.csv"
    if not f.exists():
        return {}
    with f.open() as fh:
        return {row[0].strip(): row[1].strip().lower() for row in csv.reader(fh) if len(row) >= 2}


def _summary(report: dict) -> dict:
    p = report["permit"]
    qr = p.get("qr_codes") or []
    qr_note = "-"
    if qr:
        q = qr[0]
        qr_note = "yes" + (f" (listing {q['listing_ref']})" if q.get("listing_ref") else "") + (" signed" if q.get("signed") else "")
    return {
        "verdict": report["status"],
        "agency": report.get("agency", "-"),
        "violations": ", ".join(v["code"] for v in report["violations"]) or "-",
        "printed permit": ", ".join(p["found_numbers"]) or "-",
        "permit QR": qr_note,
        "watermarks": str(report["watermarks_detected"]),
        "own logo": str(len(report.get("own_branding") or [])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--claimed", default=None, help="permit number the ad claims (optional)")
    ap.add_argument("--one-listing", action="store_true", help="treat all images as one listing")
    args = ap.parse_args()

    folder = Path(args.folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)
    listings = [(d.name + "/", sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXT))
                for d in sorted(folder.iterdir()) if d.is_dir()]
    listings = [(name, paths) for name, paths in listings if paths]
    if not images and not listings:
        raise SystemExit(f"No images in {folder}")
    expected = _expected(folder)

    with tempfile.TemporaryDirectory() as tmp:  # fresh duplicate index, so earlier runs don't count
        pipeline = CompliancePipeline(ROOT / "models" / "watermark_yolov8n.onnx",
                                      dup_index=DuplicatePhotoIndex(str(Path(tmp) / "idx.sqlite")))
        if args.one_listing:
            groups = [("all images", images + [p for _, paths in listings for p in paths])]
        else:
            groups = listings + [(p.name, [p]) for p in images]
        rows = []
        for i, (name, paths) in enumerate(groups):
            photos = [str(p) for p in paths if p.suffix.lower() != ".pdf"]
            text, qrs, reader = "", [], ""
            for j, pdf in enumerate(p for p in paths if p.suffix.lower() == ".pdf"):
                page = extract_pdf(pdf, Path(tmp) / f"pdf_{i}_{j}")
                photos += page.photo_paths
                text += "\n" + page.text
                qrs += page.qrs
                reader = page.reader
            ref = re.search(r"(\d{6,})", name)
            agency = regulatory_facts(text).get("agency") if text else None
            report = pipeline.check_listing(ListingBundle(
                listing_id=f"REAL-{i}", agent_id=f"REAL-AGENT-{i}", image_paths=photos,
                claimed_permit_number=args.claimed, page_text=text, page_qrs=qrs,
                link_listing_ref=ref.group(1) if ref else None, agency_name=agency)).to_dict()
            if text:
                facts = regulatory_facts(text)
                report["agency"] = facts.get("agency", "-")
            row = {"listing": name, "files": str(len(paths))} | _summary(report)
            if expected:
                exp = expected.get(name) or expected.get(name.rstrip("/"))
                row["expected"] = exp or "?"
                row["check"] = "?" if not exp else ("OK" if exp == report["status"] else "WRONG")
            rows.append(row)

    cols = list(rows[0])
    widths = {c: min(max(len(c), *(len(r[c]) for r in rows)), 60) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(r[c][:widths[c]].ljust(widths[c]) for c in cols))
    if expected:
        scored = [r for r in rows if r["check"] != "?"]
        ok = sum(r["check"] == "OK" for r in scored)
        print(f"\n{ok}/{len(scored)} match your expected verdicts")


if __name__ == "__main__":
    main()
