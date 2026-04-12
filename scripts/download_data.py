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
    """Download ASSIST09 dataset from Google Drive via gdown."""
    print(f"\n{'='*50}")
    print(f"  ASSIST09 Dataset")
    print(f"{'='*50}")

    dest_dir = "data/raw/assist09"
    dest_file = os.path.join(dest_dir, "skill_builder_data_corrected.csv")

    if os.path.exists(dest_file):
        print(f"  Already exists: {dest_file}")
        return True

    os.makedirs(dest_dir, exist_ok=True)

    # Try multiple sources
    urls = [
        "https://raw.githubusercontent.com/hcnoh/knowledge-tracing-collection-pytorch/master/data/ASSISTments2009/skill_builder_data_corrected.csv",
        "https://raw.githubusercontent.com/theophilee/learner-performance-prediction/master/data/assistments09/skill_builder_data_corrected.csv",
    ]

    for url in urls:
        try:
            print(f"  Downloading...")
            urllib.request.urlretrieve(url, dest_file)
            if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1e6:
                print(f"  Saved: {dest_file} ({os.path.getsize(dest_file) / 1e6:.1f} MB)")
                return True
            else:
                os.remove(dest_file)
        except Exception as e:
            print(f"  Source failed: {e}")
            continue

    print(f"  Auto-download failed. Manual download:")
    print(f"  1. Go to: https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data")
    print(f"  2. Download 'skill_builder_data_corrected.csv'")
    print(f"  3. Place it in: {dest_dir}/")
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

    os.makedirs(dest_dir, exist_ok=True)

    # Try Kaggle download
    try:
        print(f"  Attempting download from Kaggle...")
        print(f"  If this fails, download manually from:")
        print(f"    https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy")
        print(f"  Or from PSLC DataShop:")
        print(f"    https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198")
        print(f"")
        print(f"  Place these files in {dest_dir}/:")
        print(f"    - junyi_ProblemLog_original.csv")
        print(f"    - junyi_Exercise_table.csv")
    except Exception as e:
        print(f"  Error: {e}")

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
