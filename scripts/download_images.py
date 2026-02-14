"""Download galaxy cutout images from Zenodo for the Euclid Q1 dataset.

Downloads a tar archive from Zenodo and extracts the JPG cutouts into
the expected directory structure: data/raw/images/<tile>/<filename>.jpg

Usage:
    python scripts/download_images.py                          # VIS+Y (default)
    python scripts/download_images.py --type gz_arcsinh_vis_only  # VIS only
    python scripts/download_images.py --keep-tar               # Don't delete tar
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Zenodo direct download URLs for each image type
ZENODO_URLS = {
    "gz_arcsinh_vis_y": (
        "https://zenodo.org/api/records/15020547/files/"
        "cutouts_jpg_gz_arcsinh_vis_y.tar/content"
    ),
    "gz_arcsinh_vis_only": (
        "https://zenodo.org/api/records/15020547/files/"
        "cutouts_jpg_gz_arcsinh_vis_only.tar/content"
    ),
}

TAR_SIZES = {
    "gz_arcsinh_vis_y": "3.82 GB",
    "gz_arcsinh_vis_only": "3.65 GB",
}


def download_tar(url: str, output_path: Path) -> None:
    """Download a file from URL with progress reporting."""
    print(f"Downloading to: {output_path}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(
                    f"\r  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.1f}%)",
                    end="", flush=True,
                )

    print(f"\n  Download complete: {output_path.stat().st_size / 1e9:.2f} GB")


def extract_tar(tar_path: Path, output_dir: Path) -> int:
    """Extract all JPG files from a tar archive, preserving directory structure.

    Returns the number of files extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    print(f"Extracting to: {output_dir}")
    with tarfile.open(tar_path, "r") as tf:
        members = tf.getmembers()
        total = len(members)

        for i, member in enumerate(members):
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue

            tf.extract(member, path=output_dir)
            count += 1

            if count % 5000 == 0:
                print(f"\r  Extracted {count:,} images ({i+1}/{total} entries)...",
                      end="", flush=True)

    print(f"\n  Extracted {count:,} images total")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download Euclid Q1 galaxy cutout images from Zenodo"
    )
    parser.add_argument(
        "--type", choices=list(ZENODO_URLS.keys()),
        default="gz_arcsinh_vis_y",
        help="Image type to download (default: gz_arcsinh_vis_y)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "images",
        help="Directory to extract images into",
    )
    parser.add_argument(
        "--keep-tar", action="store_true",
        help="Keep the tar file after extraction (default: delete to save space)",
    )
    args = parser.parse_args()

    url = ZENODO_URLS[args.type]
    tar_path = PROJECT_ROOT / "data" / "raw" / f"cutouts_jpg_{args.type}.tar"

    print(f"Image type: {args.type}")
    print(f"Expected size: {TAR_SIZES[args.type]}")
    print()

    # Check if images already exist
    if args.output_dir.exists() and any(args.output_dir.rglob("*.jpg")):
        n_existing = sum(1 for _ in args.output_dir.rglob("*.jpg"))
        print(f"Found {n_existing:,} existing images in {args.output_dir}")
        print("Delete the directory manually to re-download.")
        return

    # Download
    if tar_path.exists():
        print(f"Tar already exists: {tar_path}")
    else:
        download_tar(url, tar_path)

    # Extract
    n_extracted = extract_tar(tar_path, args.output_dir)

    # Cleanup
    if not args.keep_tar and tar_path.exists():
        print(f"Removing tar to save space: {tar_path}")
        tar_path.unlink()
        print(f"  Freed {tar_path.stat().st_size / 1e9:.2f} GB" if tar_path.exists() else "  Done")

    print(f"\nComplete! {n_extracted:,} images in {args.output_dir}")


if __name__ == "__main__":
    main()
