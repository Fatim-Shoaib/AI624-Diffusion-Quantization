#!/usr/bin/env python3
"""
=============================================================================
Download COCO Reference Data for FID Calculation
=============================================================================

This script downloads and prepares MS-COCO validation images for use as
reference data in FID calculation.

Usage:
    python download_coco_reference.py --output-dir ./reference_data --num-images 5000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
import zipfile
import urllib.request
import shutil

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


COCO_URLS = {
    "annotations_2017": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "val_2017": "http://images.cocodataset.org/zips/val2017.zip",
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file."""
    logger.info(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


def load_coco_captions(annotations_path: Path) -> List[dict]:
    """Load COCO captions from annotations file."""
    with open(annotations_path) as f:
        data = json.load(f)
    
    # Create image_id to filename mapping
    id_to_file = {img['id']: img['file_name'] for img in data['images']}
    
    # Collect captions with image info
    captions = []
    for ann in data['annotations']:
        captions.append({
            'image_id': ann['image_id'],
            'file_name': id_to_file[ann['image_id']],
            'caption': ann['caption'],
        })
    
    return captions


def save_captions_file(
    captions: List[dict],
    output_path: Path,
    num_captions: int,
) -> None:
    """Save captions to a text file (one per line)."""
    # Get unique captions (one per image)
    seen_images = set()
    unique_captions = []
    
    for cap in captions:
        if cap['image_id'] not in seen_images:
            seen_images.add(cap['image_id'])
            unique_captions.append(cap)
        
        if len(unique_captions) >= num_captions:
            break
    
    # Save captions
    with open(output_path, 'w') as f:
        for cap in unique_captions:
            f.write(cap['caption'] + '\n')
    
    # Save full info as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(unique_captions, f, indent=2)
    
    logger.info(f"Saved {len(unique_captions)} captions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download COCO reference data for FID calculation",
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./reference_data",
        help="Output directory for reference data"
    )
    parser.add_argument(
        "--num-images", type=int, default=5000,
        help="Number of reference images to use (default: 5000)"
    )
    parser.add_argument(
        "--download-images", action="store_true",
        help="Download actual COCO images (requires ~1GB)"
    )
    parser.add_argument(
        "--captions-only", action="store_true",
        help="Only download and extract captions"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("COCO Reference Data Download")
    logger.info("=" * 60)
    
    # =========================================================================
    # Download Annotations
    # =========================================================================
    annotations_zip = output_dir / "annotations_trainval2017.zip"
    annotations_dir = output_dir / "annotations"
    
    if not annotations_dir.exists():
        if not annotations_zip.exists():
            logger.info("Downloading COCO 2017 annotations...")
            download_file(COCO_URLS["annotations_2017"], annotations_zip)
        
        extract_zip(annotations_zip, output_dir)
    
    # Load and save captions
    captions_file = annotations_dir / "captions_val2017.json"
    if captions_file.exists():
        logger.info("Loading captions...")
        captions = load_coco_captions(captions_file)
        
        captions_output = output_dir / "coco_captions.txt"
        save_captions_file(captions, captions_output, args.num_images)
    
    # =========================================================================
    # Download Images (if requested)
    # =========================================================================
    if args.download_images and not args.captions_only:
        images_zip = output_dir / "val2017.zip"
        images_dir = output_dir / "val2017"
        
        if not images_dir.exists():
            if not images_zip.exists():
                logger.info("Downloading COCO 2017 validation images (~1GB)...")
                download_file(COCO_URLS["val_2017"], images_zip)
            
            extract_zip(images_zip, output_dir)
        
        # Count images
        num_images = len(list(images_dir.glob("*.jpg")))
        logger.info(f"Downloaded {num_images} images to {images_dir}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Download Complete!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Captions file: {output_dir / 'coco_captions.txt'}")
    
    if args.download_images:
        logger.info(f"Images directory: {output_dir / 'val2017'}")
    
    logger.info("\nTo use for FID calculation:")
    logger.info(f"  --reference-path {output_dir / 'val2017'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
