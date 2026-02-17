"""Download the Euclid Q1 morphology catalog from Zenodo."""

import requests
from pathlib import Path
import sys

ZENODO_RECORD = "15020547"
CATALOG_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/morphology_catalogue.parquet/content"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "morphology_catalogue.parquet"


def download_catalog():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        size_mb = OUTPUT_FILE.stat().st_size / 1e6
        print(f"Catalog already exists: {OUTPUT_FILE} ({size_mb:.1f} MB)")
        print("Delete it manually to re-download.")
        return OUTPUT_FILE

    print(f"Downloading Euclid Q1 morphology catalog (~97 MB)...")
    print(f"Source: Zenodo record {ZENODO_RECORD}")

    response = requests.get(CATALOG_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(OUTPUT_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"\r  {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.0f}%)", end="")

    print(f"\nSaved to: {OUTPUT_FILE}")
    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        download_catalog()
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
