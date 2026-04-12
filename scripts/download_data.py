#!/usr/bin/env python3
"""
Download raw datasets for EvoLearning experiments.

Usage:
  python scripts/download_data.py --dataset assist09
  python scripts/download_data.py --dataset junyi
  python scripts/download_data.py --dataset all
"""
import os, sys, argparse, zipfile, urllib.request, shutil


def download_file(url, dest, desc=""):
    """Download with progress bar."""
    print(f"  Downloading {desc}...")
    print(f"    URL: {url}")
    print(f"    To:  {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size if total_size > 0 else 0
        print(f"\r    {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\r    Done ({os.path.getsize(dest) / 1e6:.1f} MB)")


def download_assist09():
    """Download ASSIST09 dataset."""
    print(f"\n{'='*50}")
    print(f"  ASSIST09 Dataset")
    print(f"{'='*50}")

    dest_dir = "data/raw/assist09"
    dest_file = os.path.join(dest_dir, "skill_builder_data_corrected.csv")

    if os.path.exists(dest_file):
        print(f"  Already exists: {dest_file}")
        return True

    # ASSISTments 2009-2010 data
    # Source: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data
    url = "https://drive.google.com/uc?export=download&id=1NNXHFRxcAVURFRacMHoNEZPMxgbLpMnS"

    print(f"\n  ASSIST09 requires manual download:")
    print(f"  1. Go to: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data")
    print(f"  2. Download 'skill_builder_data_corrected.csv'")
    print(f"  3. Place it in: {dest_dir}/")
    print(f"\n  Or if you have the file elsewhere:")
    print(f"    mkdir -p {dest_dir}")
    print(f"    cp /path/to/skill_builder_data_corrected.csv {dest_dir}/")

    os.makedirs(dest_dir, exist_ok=True)
    return False


def download_junyi():
    """Download Junyi Academy dataset."""
    print(f"\n{'='*50}")
    print(f"  Junyi Academy Dataset")
    print(f"{'='*50}")

    dest_dir = "data/raw/junyi/junyi_extracted"
    dest_file = os.path.join(dest_dir, "junyi_ProblemLog_original.csv")

    if os.path.exists(dest_file):
        print(f"  Already exists: {dest_file}")
        return True

    print(f"\n  Junyi requires manual download:")
    print(f"  1. Go to: https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198")
    print(f"  2. Request access and download the dataset")
    print(f"  3. Extract and place files in: {dest_dir}/")
    print(f"     Required files:")
    print(f"       - junyi_ProblemLog_original.csv")
    print(f"       - junyi_Exercise_table.csv")

    os.makedirs(dest_dir, exist_ok=True)
    return False


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download datasets")
    p.add_argument("--dataset", default="all", choices=["assist09", "junyi", "all"])
    args = p.parse_args()

    print(f"EvoLearning — Data Download")

    if args.dataset in ("assist09", "all"):
        download_assist09()
    if args.dataset in ("junyi", "all"):
        download_junyi()

    print(f"\nAfter downloading, run:")
    print(f"  python scripts/setup_pipeline.py --dataset all --gpu 0")
