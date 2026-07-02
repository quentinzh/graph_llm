#!/usr/bin/env python
"""Build item.json for TripAdvisor_corsa_filtered from OriginalReviews.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent


def build_tripadvisor_item_meta(data_dir: Path, dataset_name: str = "TripAdvisor_corsa_filtered") -> Path:
    dataset_dir = data_dir / dataset_name
    reviews_path = dataset_dir / "OriginalReviews.json"
    output_path = dataset_dir / "item.json"
    if not reviews_path.is_file():
        raise FileNotFoundError(f"Missing {reviews_path}")

    rows = json.load(reviews_path.open("r", encoding="utf-8"))
    items = {}
    for row in rows:
        item_id = str(row.get("hotelID", "")).strip()
        if not item_id:
            continue
        if item_id in items:
            continue
        title = str(row.get("hotelTitle", "")).strip() or item_id
        city = str(row.get("hotelCity", "")).strip() or "Unknown"
        heading = str(row.get("reviewHeading", "")).strip()
        items[item_id] = {
            "item": item_id,
            "title": title,
            "categories": [[city]],
            "description": heading,
        }

    output = list(items.values())
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"Wrote {len(output)} items to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Synthesize TripAdvisor item.json")
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data"),
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default="TripAdvisor_corsa_filtered",
        type=str,
    )
    args = parser.parse_args()
    build_tripadvisor_item_meta(Path(args.data_dir), args.dataset_name)


if __name__ == "__main__":
    main()
