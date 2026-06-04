"""Stage 2 - Data ingestion & protection.

Downloads the UCC 5G dataset zip and extracts it into ``data/raw/``. That
directory is gitignored, so raw data is never committed (only this script is).
``data/raw`` (immutable source) and ``data/processed`` (derived) are kept
strictly separate; nothing here writes to ``data/processed``.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import requests

from sqc import config


def _download_zip(dest: Path) -> None:
    """Stream the dataset zip to ``dest``."""
    print(f"Downloading {config.DATA_URL} ...")
    with requests.get(config.DATA_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
    print(f"Saved zip to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _git_fallback() -> Path:
    """If the raw download is blocked, shallow-clone and locate the zip."""
    clone_dir = config.RAW_DIR / "_5Gdataset_clone"
    if not clone_dir.exists():
        print(f"Direct download failed; cloning {config.GIT_FALLBACK} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", config.GIT_FALLBACK, str(clone_dir)],
            check=True,
        )
    return clone_dir / config.ZIP_NAME


def _extract(zip_path: Path) -> None:
    """Extract the zip into ``data/raw``, skipping the __MACOSX junk folder."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if config.JUNK_DIR_MARKER not in m]
        zf.extractall(config.RAW_DIR, members=members)
    print(f"Extracted dataset to {config.EXTRACTED_DIR}")


def find_csv_files() -> list[Path]:
    """Return all real measurement CSVs (excludes the __MACOSX junk folder)."""
    if not config.EXTRACTED_DIR.exists():
        return []
    return sorted(
        p
        for p in config.EXTRACTED_DIR.rglob("*.csv")
        if config.JUNK_DIR_MARKER not in str(p)
    )


def fetch_dataset(force: bool = False) -> Path:
    """Ensure the dataset is present under ``data/raw``.

    Idempotent: if already extracted and ``force`` is False, does nothing.
    Returns the path to the extracted dataset folder.
    """
    if config.EXTRACTED_DIR.exists() and not force:
        print(f"Dataset already present at {config.EXTRACTED_DIR} (use force=True to refetch)")
        return config.EXTRACTED_DIR

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = config.RAW_DIR / config.ZIP_NAME
    try:
        _download_zip(zip_path)
    except (requests.RequestException, OSError) as exc:
        print(f"Download error: {exc}")
        zip_path = _git_fallback()

    _extract(zip_path)
    return config.EXTRACTED_DIR


if __name__ == "__main__":
    path = fetch_dataset()
    print(f"{len(find_csv_files())} CSV files available under {path}")
