#!/usr/bin/env python3
"""
EdNet KT3 full pipeline: preprocess → train KT → run experiment.

Usage:
    python scripts/run_ednet_pipeline.py                    # all stages, mock
    python scripts/run_ednet_pipeline.py --provider openai  # with LLM
    python scripts/run_ednet_pipeline.py --stage preprocess  # single stage
"""

import sys, os, argparse, pickle, json, warnings
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", choices=["preprocess", "train", "run", "all"])
    p.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic", "both"])
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--max_students", type=int, default=10000,
                   help="Cap students for memory (EdNet has 297K)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-ablation", dest="no_ablation", action="store_true")
    return p.parse_args()


def stage_preprocess(args):
    """Preprocess EdNet KT3 from HuggingFace."""
    out_file = Path("data/processed/ednet/ednet_processed.pkl")
    if out_file.exists():
        print(f"[EdNet] Already processed: {out_file}")
        return

    from datasets import load_dataset

    print("[EdNet] Loading from HuggingFace (89M rows)...")
    ds = load_dataset("mgor/EDNet", "kt3", cache_dir="data/raw/ednet/hf_cache")
    t = ds["train"]

    # Convert to pandas efficiently — only respond actions
    print("[EdNet] Filtering 'respond' actions...")
    # Process in chunks to avoid OOM
    chunk_size = 5_000_000
    dfs = []
    total = len(t)

    for start in tqdm(range(0, total, chunk_size), desc="Loading chunks"):
        end = min(start + chunk_size, total)
        chunk = t[start:end]

        # Filter respond actions
        mask = [a == "respond" for a in chunk["action_type"]]
        if not any(mask):
            continue

        df_chunk = pd.DataFrame({
            "user_id": [chunk["subject_id"][i] for i, m in enumerate(mask) if m],
            "question_id": [chunk["item_id"][i] for i, m in enumerate(mask) if m],
            "correct": [int(chunk["is_correct"][i]) for i, m in enumerate(mask) if m],
            "timestamp": [chunk["timestamp"][i] / 1000 for i, m in enumerate(mask) if m],
            "user_answer": [chunk["user_answer"][i] for i, m in enumerate(mask) if m],
            "correct_answer": [chunk["correct_answer"][i] for i, m in enumerate(mask) if m],
        })
        # Keep only q-prefix items (actual questions, not bundles/explanations)
        df_chunk = df_chunk[df_chunk["question_id"].str.startswith("q")]
        dfs.append(df_chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[EdNet] Respond interactions: {len(df):,}")
    print(f"[EdNet] Unique students: {df['user_id'].nunique():,}")
    print(f"[EdNet] Unique questions: {df['question_id'].nunique():,}")

    # Load question metadata (tags) — try to get from HF dataset or separate source
    # For now, extract tags from question_id prefix + derive skill from interaction patterns
    # EdNet questions.csv has tags field with skill IDs separated by ;
    questions_meta = _get_ednet_question_tags(df)

    # Add tags column
    df["tags"] = df["question_id"].map(
        lambda qid: ";".join(questions_meta.get(qid, {}).get("kc_ids", ["unknown"])))
    df["elapsed_time"] = 30.0  # Not available in KT3 HF format directly

    # Filter students
    counts = df.groupby("user_id").size()
    valid = counts[(counts >= 20) & (counts <= 5000)].index
    # Cap for memory
    if len(valid) > args.max_students:
        np.random.seed(args.seed)
        valid = np.random.choice(valid, args.max_students, replace=False)
    df = df[df["user_id"].isin(valid)].copy()
    print(f"[EdNet] After filtering: {len(valid):,} students, {len(df):,} interactions")

    # Use common preprocess
    from simpath.data.preprocess import _common_preprocess, _print_stats
    result = _common_preprocess(df, questions_meta, "ednet",
                                min_interactions=20, max_interactions=5000)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[EdNet] Saved: {out_file}")
    _print_stats(result)


def _get_ednet_question_tags(df):
    """
    Derive skill tags for EdNet questions.
    Since HuggingFace EdNet doesn't include questions.csv tags,
    we cluster questions by co-occurrence patterns to create KC groups.
    """
    # Try loading questions.csv if available
    for path in [
        "data/raw/ednet/questions.csv",
        "data/raw/ednet/contents/questions.csv",
        "data/raw/ednet-contents/questions.csv",
    ]:
        if os.path.exists(path):
            qdf = pd.read_csv(path)
            meta = {}
            for _, row in qdf.iterrows():
                qid = str(row.get("question_id", row.get("item_id", "")))
                if not qid.startswith("q"):
                    qid = f"q{qid}"
                tags = str(row.get("tags", ""))
                kc_ids = [t.strip() for t in tags.split(";") if t.strip()] if tags else []
                meta[qid] = {"question_id": qid, "kc_ids": kc_ids}
            print(f"[EdNet] Loaded question tags from {path}: {len(meta)} questions")
            return meta

    # Fallback: assign KCs based on question difficulty bins + part number proxy
    print("[EdNet] No questions.csv found. Creating KC mapping from difficulty bins...")
    q_stats = df.groupby("question_id").agg(
        accuracy=("correct", "mean"),
        count=("correct", "count"),
    )

    # Create 50 skill bins based on accuracy patterns
    q_stats["skill_bin"] = pd.qcut(q_stats["accuracy"], q=50, labels=False, duplicates="drop")

    meta = {}
    for qid, row in q_stats.iterrows():
        meta[qid] = {
            "question_id": qid,
            "kc_ids": [f"skill_{int(row['skill_bin'])}"],
        }
    print(f"[EdNet] Created {q_stats['skill_bin'].nunique()} KC bins for {len(meta)} questions")
    return meta


def stage_train(args):
    """Train KT models on EdNet."""
    from simpath.data.dataset import KTDataset
    from simpath.kt.dkt import DKT
    from simpath.kt.akt import AKT
    from simpath.kt.saint import SAINT
    from simpath.kt.train import train_kt_model

    with open("data/processed/ednet/ednet_processed.pkl", "rb") as f:
        data = pickle.load(f)

    n_q, n_kc = data["n_questions"], data["n_kcs"]
    print(f"[Train] EdNet: Q={n_q}, KC={n_kc}, Students={len(data['students'])}")

    train_ds = KTDataset(data["students"], "train", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    val_ds = KTDataset(data["students"], "val", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    print(f"  Train seqs: {len(train_ds)}, Val seqs: {len(val_ds)}")

    # DKT (proposal: hidden=256, layers=2)
    print(f"\n{'='*50}\nDKT\n{'='*50}")
    dkt = DKT(n_q, n_kc, hidden_dim=256, num_layers=2, dropout=0.1)
    dkt, _ = train_kt_model(dkt, train_ds, val_ds, "dkt", "ednet",
                             lr=1e-3, batch_size=64, epochs=100, patience=10, device=args.device)

    # AKT (proposal: d=256, heads=8, blocks=4)
    print(f"\n{'='*50}\nAKT\n{'='*50}")
    akt = AKT(n_q, n_kc, d_model=256, num_heads=8, num_blocks=4, dropout=0.1)
    akt, _ = train_kt_model(akt, train_ds, val_ds, "akt", "ednet",
                             lr=1e-3, batch_size=64, epochs=100, patience=10, device=args.device)

    # SAINT (proposal: d=256, heads=8, enc=4, dec=4)
    print(f"\n{'='*50}\nSAINT\n{'='*50}")
    saint = SAINT(n_q, n_kc, d_model=256, num_heads=8, enc_layers=4, dec_layers=4, dropout=0.1)
    saint, _ = train_kt_model(saint, train_ds, val_ds, "saint", "ednet",
                               lr=1e-3, batch_size=64, epochs=100, patience=10, device=args.device)


def stage_run(args):
    """Run SimPath experiment on EdNet."""
    from simpath.utils.seeds import set_global_seed
    from simpath.utils.config import load_config
    from simpath.personas.definitions import THETA_GRIT, THETA_FRAGILE
    from simpath.personas.realistic import extract_realistic_params
    from simpath.personas.simulation import simulate_all_personas
    from simpath.paths.retrieval import retrieve_candidate_pool
    from simpath.paths.mock_ordering import mock_generate_paths
    from simpath.selection.minimax_regret import (
        composite_scores, minimax_regret_select,
        average_select, maximin_select, hurwicz_select, weighted_minimax_select,
    )
    from simpath.evaluation.offline_metrics import evaluate_path
    from simpath.evaluation.statistical_tests import paired_comparison
    from simpath.baselines.dkt_random import recommend_random
    from simpath.baselines.dkt_rulebased import recommend_rulebased

    cfg = load_config()
    set_global_seed(args.seed)

    with open("data/processed/ednet/ednet_processed.pkl", "rb") as f:
        data = pickle.load(f)

    students = data["students"]
    questions = data["questions"]
    s2q = data["skill_to_questions"]

    eligible = [s for s in students if len(s["held_out"]) >= 5]
    test_students = eligible[-min(args.n_test, len(eligible)):]
    print(f"[Run] EdNet, Test={len(test_students)}, Provider={args.provider}")

    L = cfg["paths"]["L"]
    K = cfg["paths"]["K"]
    N_sim = cfg["personas"]["N_sim"]
    weights = (cfg["selection"]["weights"]["w1"],
               cfg["selection"]["weights"]["w2"],
               cfg["selection"]["weights"]["w3"])

    # Build generators per provider
    if args.provider == "mock":
        providers_list = [("mock", None, None)]
    elif args.provider == "both":
        providers_list = [
            ("openai", None, "gpt-5.4-2026-03-05"),
            ("anthropic", None, "claude-sonnet-4-6"),
        ]
    else:
        providers_list = [(args.provider, None, None)]

    all_results = {}
    for prov_name, prov_key, prov_model in providers_list:
        print(f"\n{'='*60}\nProvider: {prov_name}\n{'='*60}")

        if prov_name == "mock":
            def path_gen(m, pool, K=K, L=L, **kw):
                return mock_generate_paths(m, pool, K, L), \
                    {"pool_hit_rate": 1.0, "fill_rate": 0.0, "constraint_violation_rate": 0.0}
        else:
            from simpath.llm.client import LLMClient
            from simpath.paths.ordering import generate_candidate_paths_llm
            client = LLMClient(provider=prov_name, model=prov_model, api_key=prov_key)
            print(f"  LLM: {client.provider}/{client.model}")
            def path_gen(m, pool, K=K, L=L, client=client, **kw):
                return generate_candidate_paths_llm(m, pool, client, K, L, **kw)

        methods = {n: defaultdict(list) for n in
            ["SimPath (minimax)", "B1: DKT+Random", "B2: DKT+RuleBased", "B6: SimPath-NoRobust"]}
        abl_strat = {n: defaultdict(list) for n in
            ["minimax_regret", "average", "maximin", "hurwicz_0.3", "hurwicz_0.7", "weighted_minimax"]}
        llm_stats_all = []

        for idx, student in enumerate(test_students):
            if idx % 20 == 0:
                print(f"  Student {idx+1}/{len(test_students)}...")

            mastery = student["mastery"]
            sid = student["student_id"]
            theta_real = extract_realistic_params(student["features"])
            eval_kw = dict(theta_real=theta_real, weights=weights, N_sim=N_sim)

            pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
            candidate_paths, llm_stats = path_gen(mastery, pool, K=K, L=L, seed_base=42+idx)
            llm_stats_all.append(llm_stats)

            S = simulate_all_personas(mastery, candidate_paths, theta_real, sid, N_sim)
            phi = composite_scores(S, weights)
            best_idx, _, _ = minimax_regret_select(phi)

            # SimPath
            m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                              student_id=sid, **eval_kw)
            for k, v in m.items():
                methods["SimPath (minimax)"][k].append(v)

            # Strategy ablation
            for sn, si in [("minimax_regret", best_idx), ("average", average_select(phi)),
                           ("maximin", maximin_select(phi)),
                           ("hurwicz_0.3", hurwicz_select(phi, 0.3)),
                           ("hurwicz_0.7", hurwicz_select(phi, 0.7)),
                           ("weighted_minimax", weighted_minimax_select(phi))]:
                m = evaluate_path(mastery=mastery, path=candidate_paths[si],
                                  student_id=f"{sid}_{sn}", **eval_kw)
                for k, v in m.items():
                    abl_strat[sn][k].append(v)

            # Baselines
            set_global_seed(42 + idx)
            for bn, bp in [("B1: DKT+Random", recommend_random(mastery, s2q, L=L)),
                           ("B2: DKT+RuleBased", recommend_rulebased(mastery, s2q, L=L))]:
                m = evaluate_path(mastery=mastery, path=bp,
                                  student_id=f"{sid}_{bn}", **eval_kw)
                for k, v in m.items():
                    methods[bn][k].append(v)

            # B6
            m = evaluate_path(mastery=mastery, path=candidate_paths[average_select(phi)],
                              student_id=f"{sid}_b6", **eval_kw)
            for k, v in m.items():
                methods["B6: SimPath-NoRobust"][k].append(v)

        # Print
        agg_llm = {}
        for key in ["pool_hit_rate", "fill_rate", "constraint_violation_rate"]:
            vals = [s.get(key, 0) for s in llm_stats_all if s]
            agg_llm[key] = float(np.mean(vals)) if vals else 0

        print(f"\n{'='*70}")
        print(f"RESULTS — EDNET / {prov_name.upper()}")
        if agg_llm:
            print(f"  LLM: pool_hit={agg_llm.get('pool_hit_rate',0):.3f}, "
                  f"fill={agg_llm.get('fill_rate',0):.3f}, "
                  f"violation={agg_llm.get('constraint_violation_rate',0):.3f}")
        print(f"{'='*70}")
        print(f"\n{'Method':<25} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
        print("-" * 60)
        for name, res in methods.items():
            print(f"{name:<25} {np.mean(res['MG']):8.4f} {np.mean(res['SR']):8.4f} "
                  f"{np.mean(res['CE']):8.4f} {np.mean(res['RI']):8.4f}")

        print(f"\n--- Statistical Tests ---")
        sp = methods["SimPath (minimax)"]
        for metric in ["MG", "SR", "CE", "RI"]:
            print(f"\n  {metric}:")
            for bn in ["B1: DKT+Random", "B2: DKT+RuleBased", "B6: SimPath-NoRobust"]:
                t = paired_comparison(sp[metric], methods[bn][metric], n_comparisons=12)
                sig = "*" if t["significant"] else " "
                print(f"    vs {bn:<22} p={t['p_value']:.4f} d={t['cohens_d']:+.3f} {sig}")

        print(f"\n--- Selection Strategy ---")
        print(f"{'Strategy':<20} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
        print("-" * 55)
        for sn, sr in abl_strat.items():
            print(f"{sn:<20} {np.mean(sr['MG']):8.4f} {np.mean(sr['SR']):8.4f} "
                  f"{np.mean(sr['CE']):8.4f} {np.mean(sr['RI']):8.4f}")

        all_results[prov_name] = {
            "methods": {n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in methods.items()},
            "ablation_strategy": {n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in abl_strat.items()},
            "llm_stats": agg_llm,
        }

    # Save
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ednet_{args.provider}_results.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": "ednet", **all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    args = parse_args()
    if args.stage == "all":
        stage_preprocess(args)
        stage_train(args)
        stage_run(args)
    elif args.stage == "preprocess":
        stage_preprocess(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "run":
        stage_run(args)


if __name__ == "__main__":
    main()
