"""Download EdNet KT3 and ASSISTments 2015 datasets."""

import os
import sys
import subprocess
from pathlib import Path


RAW_DIR = Path("data/raw")


def download_ednet(method="huggingface"):
    """
    Download EdNet KT3.
    Methods: huggingface (parquet, recommended), kaggle, manual
    """
    dest = RAW_DIR / "ednet"
    dest.mkdir(parents=True, exist_ok=True)

    if method == "huggingface":
        print("[EdNet] Downloading from HuggingFace (parquet)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("mgor/EDNet", "kt3", cache_dir=str(dest / "hf_cache"))
            # Save to parquet for local use
            out = dest / "ednet_kt3.parquet"
            if not out.exists():
                ds["train"].to_parquet(str(out))
                print(f"[EdNet] Saved to {out}")
            else:
                print(f"[EdNet] Already exists: {out}")
            return str(out)
        except ImportError:
            print("[EdNet] 'datasets' not installed. pip install datasets")
            print("[EdNet] Falling back to kaggle...")
            method = "kaggle"

    if method == "kaggle":
        print("[EdNet] Downloading from Kaggle...")
        print("  Requires: pip install kaggle && kaggle API token in ~/.kaggle/kaggle.json")
        # EdNet KT3-4 feather format
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "anhtu96/ednet-kt34",
            "-p", str(dest), "--unzip",
        ], check=True)
        # Also get question metadata
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "anhtu96/ednet-contents",
            "-p", str(dest), "--unzip",
        ], check=True)
        return str(dest)

    # Manual fallback
    print("[EdNet] Manual download required:")
    print("  Option A: pip install datasets && python -c 'from datasets import load_dataset; load_dataset(\"mgor/EDNet\", \"kt3\")'")
    print("  Option B: kaggle datasets download -d anhtu96/ednet-kt34")
    print(f"  Place files in: {dest}/")
    return None


def download_assistments():
    """Download ASSISTments 2015 skill builder dataset."""
    dest = RAW_DIR / "assistments"
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "2015_100_skill_builders_main_problems.csv"
    if target.exists():
        print(f"[ASSISTments] Already exists: {target}")
        return str(target)

    # Google Drive file ID
    file_id = "0B_hO8cnpcIMgUGZzRnh3bHJrSjQ"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print("[ASSISTments] Downloading from Google Drive...")
    try:
        import gdown
        gdown.download(url, str(target), quiet=False)
        return str(target)
    except ImportError:
        pass

    # Fallback: wget/curl
    try:
        subprocess.run(["gdown", url, "-O", str(target)], check=True)
        return str(target)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("[ASSISTments] Manual download required:")
    print(f"  pip install gdown && gdown {url} -O {target}")
    print(f"  Or download from: https://drive.google.com/file/d/{file_id}/view")
    print(f"  Place file at: {target}")
    return None


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ednet", "assistments", "all"], default="all")
    p.add_argument("--method", default="huggingface", help="EdNet download method")
    args = p.parse_args()

    if args.dataset in ("ednet", "all"):
        download_ednet(args.method)
    if args.dataset in ("assistments", "all"):
        download_assistments()
