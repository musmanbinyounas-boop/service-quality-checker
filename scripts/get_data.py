#!/usr/bin/env python3
"""CLI wrapper for stage-2 ingestion.

Usage:
    python scripts/get_data.py            # fetch if missing
    python scripts/get_data.py --force    # refetch even if present
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when running the script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqc import ingest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the UCC 5G dataset into data/raw")
    ap.add_argument("--force", action="store_true", help="refetch even if already present")
    args = ap.parse_args()

    path = ingest.fetch_dataset(force=args.force)
    csvs = ingest.find_csv_files()
    print(f"\nReady: {path}")
    print(f"CSV files: {len(csvs)}")
    if csvs:
        print(f"Example: {csvs[0].relative_to(path)}")


if __name__ == "__main__":
    main()
