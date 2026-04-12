"""
Preprocessing pipeline for EdNet KT3 and ASSISTments 2015.
Follows proposal Section 2.2 exactly.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


PROCESSED_DIR = Path("data/processed")


# ──────────────────────────────────────────────────────────
# EdNet KT3
# ──────────────────────────────────────────────────────────

def preprocess_ednet(raw_path: str, out_dir: str = None) -> str:
    """
    Preprocess EdNet KT3.
    Input:  parquet or directory of per-student CSVs
    Output: processed pickle with students, questions, skill_to_questions
    """
    out_dir = Path(out_dir or PROCESSED_DIR / "ednet")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ednet_processed.pkl"

    if out_file.exists():
        print(f"[EdNet] Already processed: {out_file}")
        return str(out_file)

    raw_path = Path(raw_path)
    print("[EdNet] Loading raw data...")

    # Load question metadata (tags)
    questions_meta = _load_ednet_questions(raw_path)

    # Load interactions
    if raw_path.suffix == ".parquet":
        df = pd.read_parquet(raw_path)
    elif (raw_path / "ednet_kt3.parquet").exists():
        df = pd.read_parquet(raw_path / "ednet_kt3.parquet")
    else:
        # Per-student CSV directory
        df = _load_ednet_csvs(raw_path)

    print(f"[EdNet] Raw interactions: {len(df):,}")

    # Normalize columns
    df = _normalize_ednet_columns(df, questions_meta)

    # Run common preprocessing
    result = _common_preprocess(df, questions_meta, "ednet",
                                min_interactions=20, max_interactions=5000)

    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[EdNet] Saved: {out_file}")
    _print_stats(result)
    return str(out_file)


def _load_ednet_questions(raw_path: Path) -> Dict:
    """Load EdNet question metadata (tags/skills)."""
    candidates = [
        raw_path / "questions.csv",
        raw_path / "contents" / "questions.csv",
        raw_path.parent / "ednet-contents" / "questions.csv",
        raw_path.parent / "questions.csv",
    ]
    for p in candidates:
        if p.exists():
            qdf = pd.read_csv(p)
            print(f"[EdNet] Loaded question metadata: {p} ({len(qdf)} questions)")
            meta = {}
            for _, row in qdf.iterrows():
                qid = str(row.get("question_id", row.get("item_id", "")))
                tags_raw = str(row.get("tags", ""))
                tags = [t.strip() for t in tags_raw.split(";") if t.strip()] if tags_raw else []
                meta[qid] = {
                    "question_id": qid,
                    "kc_ids": tags,
                    "correct_answer": str(row.get("correct_answer", "")),
                    "part": int(row.get("part", 0)) if pd.notna(row.get("part")) else 0,
                }
            return meta

    print("[EdNet] Warning: questions.csv not found. Using item_id tags from interactions.")
    return {}


def _load_ednet_csvs(raw_path: Path) -> pd.DataFrame:
    """Load per-student CSV files (KT3 format)."""
    csv_dir = raw_path / "KT3" if (raw_path / "KT3").exists() else raw_path
    files = sorted(csv_dir.glob("*.csv"))[:50000]  # cap for memory
    print(f"[EdNet] Loading {len(files)} student CSV files...")

    dfs = []
    for f in tqdm(files, desc="Loading CSVs"):
        uid = f.stem
        sdf = pd.read_csv(f)
        sdf["user_id"] = uid
        dfs.append(sdf)

    return pd.concat(dfs, ignore_index=True)


def _normalize_ednet_columns(df: pd.DataFrame, questions_meta: Dict) -> pd.DataFrame:
    """Normalize EdNet columns to common format."""
    col_map = {}

    # Detect column names (different sources have different names)
    for orig, target in [
        (["user_id", "userId"], "user_id"),
        (["item_id", "question_id", "itemId"], "question_id"),
        (["timestamp", "time"], "timestamp"),
        (["is_correct", "correct", "user_answer"], "correct"),
        (["elapsed_time", "elapsed", "ms_first_response"], "elapsed_time"),
        (["tags", "skill_id", "subject_id"], "tags"),
    ]:
        for c in orig:
            if c in df.columns:
                col_map[c] = target
                break

    df = df.rename(columns=col_map)

    # Ensure user_id is string
    df["user_id"] = df["user_id"].astype(str)

    # Ensure question_id is string
    if "question_id" in df.columns:
        df["question_id"] = df["question_id"].astype(str)

    # Handle correctness
    if "correct" in df.columns:
        if df["correct"].dtype == object:
            # user_answer format: compare to correct_answer
            if questions_meta:
                df["correct"] = df.apply(
                    lambda r: int(str(r["correct"]) == questions_meta.get(
                        str(r.get("question_id", "")), {}).get("correct_answer", "")),
                    axis=1,
                )
            else:
                df["correct"] = df["correct"].apply(lambda x: int(x) if str(x).isdigit() else 0)
        df["correct"] = df["correct"].astype(int).clip(0, 1)

    # Handle timestamp (ms to seconds if needed)
    if "timestamp" in df.columns:
        if df["timestamp"].median() > 1e12:
            df["timestamp"] = df["timestamp"] / 1000

    # Handle elapsed_time
    if "elapsed_time" in df.columns:
        df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce").fillna(30)
        if df["elapsed_time"].median() > 1000:
            df["elapsed_time"] = df["elapsed_time"] / 1000  # ms to s
    else:
        df["elapsed_time"] = 30.0

    # Handle tags/skills
    if "tags" not in df.columns:
        if questions_meta:
            df["tags"] = df["question_id"].apply(
                lambda qid: ";".join(questions_meta.get(str(qid), {}).get("kc_ids", [])))
        else:
            df["tags"] = ""

    return df


# ──────────────────────────────────────────────────────────
# ASSISTments 2015
# ──────────────────────────────────────────────────────────

def preprocess_assistments(raw_path: str, out_dir: str = None) -> str:
    """
    Preprocess ASSISTments 2015.
    Input:  CSV file
    Output: processed pickle
    """
    out_dir = Path(out_dir or PROCESSED_DIR / "assistments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "assistments_processed.pkl"

    if out_file.exists():
        print(f"[ASSISTments] Already processed: {out_file}")
        return str(out_file)

    print("[ASSISTments] Loading raw data...")
    df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    print(f"[ASSISTments] Raw rows: {len(df):,}")

    # Normalize columns
    df = _normalize_assistments_columns(df)

    # Build questions metadata from the data itself
    questions_meta = _build_assistments_questions(df)

    result = _common_preprocess(df, questions_meta, "assistments",
                                min_interactions=20, max_interactions=5000)

    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[ASSISTments] Saved: {out_file}")
    _print_stats(result)
    return str(out_file)


def _normalize_assistments_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ASSISTments 2015 columns.
    Raw columns: user_id, log_id, sequence_id, correct
    - sequence_id = skill builder (100 total) → acts as both question_id and KC
    - log_id = unique interaction ID → acts as timestamp proxy (sequential)
    """
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)

    # sequence_id = skill builder → question_id and KC tag
    df["question_id"] = df["sequence_id"].astype(str)
    df["tags"] = df["sequence_id"].astype(str)

    # correct → binary
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0)
    df["correct"] = (df["correct"] >= 1).astype(int)

    # log_id → timestamp proxy (monotonically increasing per student)
    df["timestamp"] = pd.to_numeric(df["log_id"], errors="coerce").fillna(0).astype(float)

    # No elapsed_time available
    df["elapsed_time"] = 30.0

    return df


def _build_assistments_questions(df: pd.DataFrame) -> Dict:
    """Build question metadata from ASSISTments interaction data."""
    meta = {}
    for qid, grp in df.groupby("question_id"):
        tags_raw = grp["tags"].dropna().unique()
        all_tags = set()
        for t in tags_raw:
            for tag in str(t).split(";"):
                tag = tag.strip()
                if tag and tag != "nan":
                    all_tags.add(tag)

        meta[str(qid)] = {
            "question_id": str(qid),
            "kc_ids": list(all_tags) if all_tags else ["unknown"],
        }
    return meta


def preprocess_assist09(raw_path: str, out_dir: str = None) -> str:
    """
    Preprocess ASSISTments 2009-2010 (skill_builder_data_corrected.csv).
    Columns: order_id, user_id, skill_id, skill_name, correct, ...
    """
    out_dir = Path(out_dir or PROCESSED_DIR / "assist09")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "assist09_processed.pkl"

    if out_file.exists():
        print(f"[ASSIST09] Already processed: {out_file}")
        return str(out_file)

    print("[ASSIST09] Loading raw data...")
    df = pd.read_csv(raw_path, encoding="latin-1", low_memory=False)
    print(f"[ASSIST09] Raw rows: {len(df):,}")

    # Drop rows without skill_id
    df = df.dropna(subset=["skill_id"])
    df["skill_id"] = df["skill_id"].astype(int)

    # Normalize columns
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["question_id"] = df["problem_id"].astype(str)
    df["tags"] = df["skill_id"].astype(str)
    df["correct"] = df["correct"].clip(0, 1).astype(int)
    df["timestamp"] = pd.to_numeric(df["order_id"], errors="coerce").fillna(0).astype(float)
    df["elapsed_time"] = pd.to_numeric(df["ms_first_response"], errors="coerce").fillna(30000) / 1000.0
    df["elapsed_time"] = df["elapsed_time"].clip(1, 600)

    print(f"[ASSIST09] Students: {df['user_id'].nunique()}, "
          f"Skills: {df['tags'].nunique()}, Problems: {df['question_id'].nunique()}")

    # Build question metadata
    questions_meta = {}
    for qid, grp in df.groupby("question_id"):
        skills = grp["tags"].dropna().unique().tolist()
        questions_meta[str(qid)] = {
            "question_id": str(qid),
            "kc_ids": [str(s) for s in skills] if skills else ["unknown"],
        }

    result = _common_preprocess(df, questions_meta, "assist09",
                                min_interactions=20, max_interactions=5000)

    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[ASSIST09] Saved: {out_file}")
    _print_stats(result)
    return str(out_file)


# ──────────────────────────────────────────────────────────
# Common preprocessing (Section 2.2 Steps 1-7)
# ──────────────────────────────────────────────────────────

def _common_preprocess(
    df: pd.DataFrame,
    questions_meta: Dict,
    dataset_name: str,
    min_interactions: int = 20,
    max_interactions: int = 5000,
) -> Dict:
    """
    Common preprocessing pipeline following proposal Section 2.2.
    """
    required = ["user_id", "question_id", "correct", "timestamp", "elapsed_time", "tags"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Step 1: Filter by interaction count
    counts = df.groupby("user_id").size()
    valid_users = counts[(counts >= min_interactions) & (counts <= max_interactions)].index
    df = df[df["user_id"].isin(valid_users)].copy()
    print(f"  After filtering ({min_interactions}-{max_interactions}): "
          f"{len(valid_users):,} students, {len(df):,} interactions")

    # Step 3: Sort by timestamp within each student
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Step 4: Compute per-question difficulty
    q_stats = df.groupby("question_id")["correct"].agg(["sum", "count"])
    q_difficulty = {}
    for qid, row in q_stats.iterrows():
        q_difficulty[str(qid)] = 1.0 - (row["sum"] / max(row["count"], 1))

    # Build question metadata with difficulty
    all_questions = {}
    skill_to_questions = defaultdict(list)
    all_kcs = set()

    for qid in df["question_id"].unique():
        qid_str = str(qid)
        if qid_str in questions_meta:
            kc_ids = questions_meta[qid_str].get("kc_ids", [])
        else:
            # Extract from tags column
            tags_vals = df[df["question_id"] == qid]["tags"].dropna().unique()
            kc_ids = set()
            for t in tags_vals:
                for tag in str(t).split(";"):
                    tag = tag.strip()
                    if tag and tag != "nan":
                        kc_ids.add(tag)
            kc_ids = list(kc_ids) if kc_ids else ["unknown"]

        q = {
            "question_id": qid_str,
            "kc_ids": kc_ids,
            "difficulty": q_difficulty.get(qid_str, 0.5),
        }
        all_questions[qid_str] = q
        all_kcs.update(kc_ids)
        for kc in kc_ids:
            skill_to_questions[kc].append(q)

    # Map KC IDs to integers for model input
    kc_list = sorted(all_kcs)
    kc_to_idx = {kc: i for i, kc in enumerate(kc_list)}
    q_list = sorted(all_questions.keys())
    q_to_idx = {q: i for i, q in enumerate(q_list)}

    # Step 5: Split per student + Step 6-7: Extract features
    print("  Processing students...")
    students = []
    for uid, grp in tqdm(df.groupby("user_id"), desc="Students"):
        grp = grp.sort_values("timestamp")
        n = len(grp)
        if n < min_interactions:
            continue

        # Step 5: 70/10/10/10 split
        s1 = int(n * 0.7)
        s2 = int(n * 0.8)
        s3 = int(n * 0.9)

        interactions = []
        for _, row in grp.iterrows():
            qid = str(row["question_id"])
            q_meta = all_questions.get(qid, {"kc_ids": ["unknown"], "difficulty": 0.5})
            interactions.append({
                "question_id": qid,
                "question_idx": q_to_idx.get(qid, 0),
                "kc_ids": q_meta["kc_ids"],
                "kc_idxs": [kc_to_idx.get(kc, 0) for kc in q_meta["kc_ids"]],
                "difficulty": q_meta["difficulty"],
                "correct": int(row["correct"]),
                "elapsed_time": float(row["elapsed_time"]),
                "timestamp": float(row["timestamp"]),
            })

        train = interactions[:s1]
        val = interactions[s1:s2]
        rec_input = interactions[s2:s3]
        held_out = interactions[s3:]

        # Step 7: Behavioral features
        features = _extract_behavioral_features(interactions, kc_list)

        # Mastery estimate from rec_input
        mastery = _estimate_mastery(rec_input, train, kc_list)

        students.append({
            "student_id": str(uid),
            "train": train,
            "val": val,
            "rec_input": rec_input,
            "held_out": held_out,
            "mastery": mastery,
            "features": features,
        })

    result = {
        "dataset_name": dataset_name,
        "students": students,
        "questions": list(all_questions.values()),
        "skill_to_questions": dict(skill_to_questions),
        "kc_list": kc_list,
        "kc_to_idx": kc_to_idx,
        "q_list": q_list,
        "q_to_idx": q_to_idx,
        "n_questions": len(q_list),
        "n_kcs": len(kc_list),
    }
    return result


def _extract_behavioral_features(interactions: List[dict], kc_list: List[str]) -> Dict:
    """Step 7: Extract per-student behavioral features."""
    # Session splitting (gap > 30 min)
    sessions = [[interactions[0]]]
    for h in interactions[1:]:
        if h["timestamp"] - sessions[-1][-1]["timestamp"] > 1800:
            sessions.append([])
        sessions[-1].append(h)

    avg_session_length = np.mean([len(s) for s in sessions])

    # Dropout patterns
    dropout_rates = {}
    for k in range(1, 6):
        end_count, cont_count = 0, 0
        for session in sessions:
            consec = 0
            for i, h in enumerate(session):
                if h["correct"] == 0:
                    consec += 1
                    if consec == k:
                        if i == len(session) - 1:
                            end_count += 1
                        else:
                            cont_count += 1
                else:
                    consec = 0
        total = end_count + cont_count
        dropout_rates[k] = end_count / max(total, 1)

    # Skip rate
    fast = [h for h in interactions if h["elapsed_time"] < 5]
    skip_rate = len(fast) / max(len(interactions), 1)

    # Overall accuracy
    accuracy = np.mean([h["correct"] for h in interactions])

    # Avg elapsed time
    avg_elapsed = np.mean([h["elapsed_time"] for h in interactions])

    return {
        "avg_session_length": float(avg_session_length),
        "dropout_rates": dropout_rates,
        "avg_elapsed_time": float(avg_elapsed),
        "skip_rate": float(skip_rate),
        "overall_accuracy": float(accuracy),
    }


def _estimate_mastery(rec_input, train, kc_list):
    """Estimate per-KC mastery from recent interactions."""
    mastery = {}
    # Use rec_input first, fall back to train
    for kc in kc_list:
        rec_kc = [h["correct"] for h in rec_input if kc in h["kc_ids"]]
        if rec_kc:
            mastery[kc] = float(np.mean(rec_kc))
        else:
            train_kc = [h["correct"] for h in train if kc in h["kc_ids"]]
            mastery[kc] = float(np.mean(train_kc)) if train_kc else 0.5
    return mastery


def _print_stats(result):
    print(f"\n  Dataset: {result['dataset_name']}")
    print(f"  Students: {len(result['students']):,}")
    print(f"  Questions: {result['n_questions']:,}")
    print(f"  KCs: {result['n_kcs']:,}")
    n_train = sum(len(s["train"]) for s in result["students"])
    n_test = sum(len(s["held_out"]) for s in result["students"])
    print(f"  Train interactions: {n_train:,}")
    print(f"  Held-out interactions: {n_test:,}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ednet", "assistments", "all"], default="all")
    p.add_argument("--ednet_raw", default="data/raw/ednet")
    p.add_argument("--assist_raw", default="data/raw/assistments/2015_100_skill_builders_main_problems.csv")
    args = p.parse_args()

    if args.dataset in ("ednet", "all"):
        preprocess_ednet(args.ednet_raw)
    if args.dataset in ("assistments", "all"):
        preprocess_assistments(args.assist_raw)
