#!/usr/bin/env python3
"""Preprocess EdNet KT1 + Contents/questions.csv → processed pickle."""
import sys, os, argparse, pickle, csv, warnings
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_students", type=int, default=50000)
    p.add_argument("--min_interactions", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_file = Path("data/processed/ednet/ednet_processed.pkl")
    if out_file.exists():
        print(f"Already processed: {out_file}")
        return

    kt1_dir = Path("data/raw/ednet/KT1")
    questions_csv = Path("data/raw/ednet/contents/questions.csv")

    # ═══ 1. Load questions.csv → question_id → (correct_answer, tags) ═══
    print("[EdNet] Loading questions.csv...")
    q_meta = {}
    all_tags = set()
    with open(questions_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row['question_id']
            tags = [t.strip() for t in row['tags'].split(';') if t.strip() and t.strip() != '-1']
            q_meta[qid] = {
                'question_id': qid,
                'correct_answer': row['correct_answer'],
                'kc_ids': tags if tags else ['unknown'],
                'part': int(row.get('part', 0)),
            }
            all_tags.update(tags)
    print(f"  Questions: {len(q_meta)}, Tags: {len(all_tags)}")

    # ═══ 2. Scan KT1 student files, count interactions ═══
    print("[EdNet] Scanning student files...")
    student_files = sorted(kt1_dir.glob("u*.csv"))
    print(f"  Total student files: {len(student_files)}")

    # Count interactions per student (fast scan)
    counts = {}
    for sf in tqdm(student_files, desc="Counting"):
        uid = sf.stem
        with open(sf) as f:
            n = sum(1 for _ in f) - 1  # minus header
        if n >= args.min_interactions:
            counts[uid] = n

    print(f"  Students with >= {args.min_interactions} interactions: {len(counts)}")

    # Sample
    if len(counts) > args.max_students:
        np.random.seed(args.seed)
        selected = np.random.choice(list(counts.keys()), args.max_students, replace=False)
        selected = set(selected)
    else:
        selected = set(counts.keys())
    print(f"  Selected: {len(selected)}")

    # ═══ 3. Load selected students ═══
    print("[EdNet] Loading student interactions...")
    rows = []
    for sf in tqdm([kt1_dir / f"{uid}.csv" for uid in selected], desc="Loading"):
        uid = sf.stem
        with open(sf) as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = row['question_id']
                if qid not in q_meta:
                    continue
                ua = row['user_answer']
                ca = q_meta[qid]['correct_answer']
                correct = 1 if ua == ca else 0
                rows.append({
                    'user_id': uid,
                    'question_id': qid,
                    'correct': correct,
                    'timestamp': int(row['timestamp']),
                    'elapsed_time': min(max(int(row['elapsed_time']) / 1000, 1), 600),
                    'tags': ';'.join(q_meta[qid]['kc_ids']),
                })

    df = pd.DataFrame(rows)
    print(f"  Loaded: {len(df):,} interactions, {df['user_id'].nunique()} students")

    # ═══ 4. Common preprocessing ═══
    from simpath.data.preprocess import _common_preprocess, _print_stats
    result = _common_preprocess(df, q_meta, "ednet",
                                min_interactions=args.min_interactions,
                                max_interactions=5000)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    _print_stats(result)


if __name__ == "__main__":
    main()
