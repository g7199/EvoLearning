#!/usr/bin/env python3
"""Junyi dataset pipeline — standard dataset used by SRC/IB-GRPO."""

import sys, os, argparse, pickle, warnings
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["preprocess", "train", "all"])
    p.add_argument("--max_students", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def stage_preprocess(args):
    out_file = Path("data/processed/junyi/junyi_processed.pkl")
    if out_file.exists():
        print(f"Already processed: {out_file}")
        return

    print("[Junyi] Loading...")
    log = pd.read_csv("data/raw/junyi/junyi_extracted/junyi_ProblemLog_original.csv", low_memory=False)
    ex = pd.read_csv("data/raw/junyi/junyi_extracted/junyi_Exercise_table.csv")

    # Build exercise → topic mapping (topic = KC)
    ex_to_topic = dict(zip(ex["name"], ex["topic"].fillna("unknown")))

    # Normalize columns
    df = log[["user_id", "exercise", "correct", "time_done", "time_taken"]].copy()
    df = df.rename(columns={"exercise": "question_id"})
    df["user_id"] = df["user_id"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["correct"] = df["correct"].astype(int).clip(0, 1)
    df["tags"] = df["question_id"].map(lambda x: ex_to_topic.get(x, "unknown"))
    df["elapsed_time"] = pd.to_numeric(df["time_taken"], errors="coerce").fillna(30)
    df["elapsed_time"] = df["elapsed_time"].clip(1, 600)
    df["timestamp"] = pd.to_numeric(df["time_done"], errors="coerce")
    df.loc[df["timestamp"].isna(), "timestamp"] = range(df["timestamp"].isna().sum())

    print(f"  Rows: {len(df):,}, Students: {df['user_id'].nunique():,}, "
          f"Exercises: {df['question_id'].nunique()}, Topics: {df['tags'].nunique()}")

    # Build question metadata
    questions_meta = {}
    for qid in df["question_id"].unique():
        topic = ex_to_topic.get(qid, "unknown")
        questions_meta[qid] = {"question_id": qid, "kc_ids": [topic]}

    # Filter and sample students
    counts = df.groupby("user_id").size()
    valid = counts[(counts >= 20) & (counts <= 5000)].index
    if len(valid) > args.max_students:
        np.random.seed(args.seed)
        valid = np.random.choice(valid, args.max_students, replace=False)
    df = df[df["user_id"].isin(valid)].copy()
    print(f"  After filtering: {len(valid):,} students, {len(df):,} interactions")

    from simpath.data.preprocess import _common_preprocess, _print_stats
    result = _common_preprocess(df, questions_meta, "junyi",
                                min_interactions=20, max_interactions=5000)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    _print_stats(result)


def stage_train(args):
    from simpath.data.dataset import KTDataset
    from simpath.kt.dkt import DKT
    from simpath.kt.akt import AKT
    from simpath.kt.saint import SAINT
    from simpath.kt.train import train_kt_model

    with open("data/processed/junyi/junyi_processed.pkl", "rb") as f:
        data = pickle.load(f)

    n_q, n_kc = data["n_questions"], data["n_kcs"]
    print(f"Junyi: Q={n_q}, KC={n_kc}, Students={len(data['students'])}")

    train_ds = KTDataset(data["students"], "train", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    val_ds = KTDataset(data["students"], "val", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    print(f"Train={len(train_ds)}, Val={len(val_ds)}")

    for name, cls, kw in [
        ("dkt", DKT, dict(hidden_dim=256, num_layers=2, dropout=0.1)),
        ("akt", AKT, dict(d_model=256, num_heads=8, num_blocks=4, dropout=0.1)),
    ]:
        print(f"\n=== {name.upper()} ===")
        model = cls(n_q, n_kc, **kw)
        model, _ = train_kt_model(model, train_ds, val_ds, name, "junyi",
                                   lr=1e-3, batch_size=64, epochs=50, patience=10,
                                   device=args.device)


def main():
    args = parse_args()
    if args.stage == "all":
        stage_preprocess(args)
        stage_train(args)
    else:
        globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
