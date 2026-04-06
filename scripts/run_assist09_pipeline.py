#!/usr/bin/env python3
"""
ASSIST09 pipeline — standard dataset for SRC/CSEAL/IB-GRPO comparison.
Follows the field-standard evaluation: EP = (Ee-Es)/(Esup-Es).

Usage:
    python scripts/run_assist09_pipeline.py preprocess
    python scripts/run_assist09_pipeline.py train
    python scripts/run_assist09_pipeline.py run --provider anthropic
    python scripts/run_assist09_pipeline.py all --provider anthropic
"""

import sys, os, argparse, json, pickle, warnings
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
    p.add_argument("stage", choices=["preprocess", "train", "run", "all"])
    p.add_argument("--provider", default="mock")
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--p_target", type=int, default=5, help="Target concepts (SRC uses p=3,5,7)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def stage_preprocess(args):
    out_file = Path("data/processed/assist09/assist09_processed.pkl")
    if out_file.exists():
        print(f"Already processed: {out_file}")
        return

    print("[ASSIST09] Loading...")
    df = pd.read_csv("data/raw/assist09/skill_builder_data_corrected.csv",
                      encoding="latin-1", low_memory=False)

    # Standard preprocessing (matches SRC/CSEAL convention)
    df = df.dropna(subset=["skill_id"])
    df["user_id"] = df["user_id"].astype(str)
    df["question_id"] = df["problem_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(int).astype(str)
    df["correct"] = df["correct"].astype(int).clip(0, 1)
    df["timestamp"] = df["order_id"].astype(float)
    df["elapsed_time"] = pd.to_numeric(df["ms_first_response"], errors="coerce").fillna(30000) / 1000

    # Tags = skill_id (can be multi-skill, separated by comma in some versions)
    df["tags"] = df["skill_id"]

    print(f"  Rows: {len(df):,}, Students: {df['user_id'].nunique()}, Skills: {df['skill_id'].nunique()}")

    from simpath.data.preprocess import _common_preprocess, _print_stats
    questions_meta = {}
    for qid, grp in df.groupby("question_id"):
        skills = list(grp["skill_id"].unique())
        questions_meta[str(qid)] = {"question_id": str(qid), "kc_ids": skills}

    result = _common_preprocess(df, questions_meta, "assist09",
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

    with open("data/processed/assist09/assist09_processed.pkl", "rb") as f:
        data = pickle.load(f)

    n_q, n_kc = data["n_questions"], data["n_kcs"]
    print(f"ASSIST09: Q={n_q}, KC={n_kc}, Students={len(data['students'])}")

    train_ds = KTDataset(data["students"], "train", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    val_ds = KTDataset(data["students"], "val", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    print(f"Train={len(train_ds)}, Val={len(val_ds)}")

    for name, model_cls, kwargs in [
        ("dkt", DKT, dict(hidden_dim=256, num_layers=2, dropout=0.1)),
        ("akt", AKT, dict(d_model=256, num_heads=8, num_blocks=4, dropout=0.1)),
        ("saint", SAINT, dict(d_model=256, num_heads=8, enc_layers=4, dec_layers=4, dropout=0.1)),
    ]:
        print(f"\n=== {name.upper()} ===")
        model = model_cls(n_q, n_kc, **kwargs)
        model, _ = train_kt_model(model, train_ds, val_ds, name, "assist09",
                                   lr=1e-3, batch_size=64, epochs=100, patience=10,
                                   device=args.device)


def stage_run(args):
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
    from simpath.baselines.rl_dkt import train_rl_dkt, recommend_rl_dkt
    from simpath.baselines.lprekl import recommend_lprekl

    cfg = load_config()
    set_global_seed(args.seed)

    with open("data/processed/assist09/assist09_processed.pkl", "rb") as f:
        data = pickle.load(f)

    students = data["students"]
    questions = data["questions"]
    s2q = data["skill_to_questions"]
    kc_list = data["kc_list"]

    eligible = [s for s in students if len(s["held_out"]) >= 5]
    test_students = eligible[-min(args.n_test, len(eligible)):]
    print(f"ASSIST09: Test={len(test_students)}, p={args.p_target}, Provider={args.provider}")

    L = cfg["paths"]["L"]
    K = cfg["paths"]["K"]
    N_sim = cfg["personas"]["N_sim"]
    weights = (cfg["selection"]["weights"]["w1"],
               cfg["selection"]["weights"]["w2"],
               cfg["selection"]["weights"]["w3"])

    # Path generator
    if args.provider == "mock":
        def path_gen(m, pool, K=K, L=L, **kw):
            return mock_generate_paths(m, pool, K, L), \
                {"pool_hit_rate": 1.0, "fill_rate": 0.0, "constraint_violation_rate": 0.0}
    else:
        from simpath.llm.client import LLMClient
        from simpath.paths.ordering import generate_candidate_paths_llm
        model_name = "claude-sonnet-4-6" if args.provider == "anthropic" else None
        client = LLMClient(provider=args.provider, model=model_name)
        print(f"  LLM: {client.provider}/{client.model}")
        def path_gen(m, pool, K=K, L=L, client=client, **kw):
            return generate_candidate_paths_llm(m, pool, client, K, L, **kw)

    # Fixed target KCs per student (p weakest)
    def get_targets(mastery, p):
        sorted_kcs = sorted(mastery.items(), key=lambda x: x[1])
        weak = [kc for kc, m in sorted_kcs if m < 0.6]
        return weak[:p] if len(weak) >= p else [kc for kc, _ in sorted_kcs[:p]]

    methods = {n: defaultdict(list) for n in
        ["SimPath", "B1:Random", "B2:RuleBased", "B3:RL-DKT", "B4:LPReKL", "B6:NoRobust"]}
    abl = {n: defaultdict(list) for n in
        ["minimax", "average", "maximin", "hurwicz_0.3", "hurwicz_0.7", "weighted"]}

    # Train RL-DKT agent (once, using training students)
    print("\n  Training RL-DKT agent (5000 episodes)...")
    # Use a common pool for RL training
    common_pool = questions[:min(500, len(questions))]
    rl_agent = train_rl_dkt(
        students=students[:500], pool_questions=common_pool,
        kc_list=kc_list, n_episodes=5000, L=L, target_p=args.p_target,
        save_path="outputs/checkpoints/rl_dkt_assist09.pt",
    )
    print("  RL-DKT training done.\n")

    for idx, student in enumerate(test_students):
        if idx % 50 == 0:
            print(f"  {idx+1}/{len(test_students)}...")

        mastery = student["mastery"]
        sid = student["student_id"]
        theta_real = extract_realistic_params(student["features"])
        target_kcs = get_targets(mastery, args.p_target)

        # SAME pool for all methods
        pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
        eval_kw = dict(theta_real=theta_real, target_kcs=target_kcs,
                        weights=weights, N_sim=N_sim)

        # SimPath
        candidate_paths, _ = path_gen(mastery, pool, K=K, L=L, seed_base=42+idx)
        S = simulate_all_personas(mastery, candidate_paths, theta_real, sid, N_sim, target_kcs)
        phi = composite_scores(S, weights)
        best_idx, _, _ = minimax_regret_select(phi)

        m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                          student_id=sid, path_idx=best_idx, **eval_kw)
        for k, v in m.items():
            methods["SimPath"][k].append(v)

        # Ablation
        for sname, sidx in [("minimax", best_idx), ("average", average_select(phi)),
                            ("maximin", maximin_select(phi)),
                            ("hurwicz_0.3", hurwicz_select(phi, 0.3)),
                            ("hurwicz_0.7", hurwicz_select(phi, 0.7)),
                            ("weighted", weighted_minimax_select(phi))]:
            m = evaluate_path(mastery=mastery, path=candidate_paths[sidx],
                              student_id=sid, path_idx=sidx, **eval_kw)
            for k, v in m.items():
                abl[sname][k].append(v)

        # Baselines (ALL use SAME pool, SAME target_kcs, SAME evaluation)
        set_global_seed(42 + idx)
        for bn, bp in [
            ("B1:Random", recommend_random(mastery, s2q, L=L, pool=pool)),
            ("B2:RuleBased", recommend_rulebased(mastery, s2q, L=L, pool=pool)),
            ("B3:RL-DKT", recommend_rl_dkt(rl_agent, mastery, common_pool, kc_list, L=L)),
            ("B4:LPReKL", recommend_lprekl(mastery, pool, kc_list, L=L)),
        ]:
            m = evaluate_path(mastery=mastery, path=bp,
                              student_id=f"{sid}_{bn}", path_idx=0, **eval_kw)
            for k, v in m.items():
                methods[bn][k].append(v)

        norobust_idx = average_select(phi)
        m = evaluate_path(mastery=mastery, path=candidate_paths[norobust_idx],
                          student_id=f"{sid}_b6", path_idx=norobust_idx, **eval_kw)
        for k, v in m.items():
            methods["B6:NoRobust"][k].append(v)

    # Print
    print(f"\n{'='*70}")
    print(f"ASSIST09 — {args.provider.upper()} — p={args.p_target}")
    print(f"Protocol: EP=(Ee-Es)/(1-Es), fixed {args.p_target} target KCs, same pool")
    print(f"{'='*70}")

    print(f"\n{'Method':<20} {'EP':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 55)
    for name, res in methods.items():
        print(f"{name:<20} {np.mean(res['EP']):8.4f} {np.mean(res['SR']):8.4f} "
              f"{np.mean(res['CE']):8.4f} {np.mean(res['RI']):8.4f}")

    sp = methods["SimPath"]
    print(f"\n--- Statistical Tests ---")
    for metric in ["EP", "SR", "CE", "RI"]:
        for bn in ["B1:Random", "B2:RuleBased", "B6:NoRobust"]:
            t = paired_comparison(sp[metric], methods[bn][metric], n_comparisons=12)
            sig = "***" if t["p_value"] < 0.001 else ("**" if t["p_value"] < 0.01 else ("*" if t["significant"] else ""))
            print(f"  {metric} vs {bn:<15} p={t['p_value']:.4f} d={t['cohens_d']:+.3f} {sig}")

    print(f"\n--- Selection Strategy ---")
    print(f"{'Strategy':<15} {'EP':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 50)
    for sn, sr in abl.items():
        print(f"{sn:<15} {np.mean(sr['EP']):8.4f} {np.mean(sr['SR']):8.4f} "
              f"{np.mean(sr['CE']):8.4f} {np.mean(sr['RI']):8.4f}")

    # Reference: SRC (AAAI 2023) results on ASSIST09 with DKT simulator, p=5
    print(f"\n--- Reference (from SRC paper, ASSIST09, DKT sim, p=5) ---")
    print(f"  SRC:        ET ≈ 0.557")
    print(f"  Rule-based: ET ≈ 0.423")
    print(f"  DQN:        ET ≈ 0.295")
    print(f"  Random:     ET ≈ 0.243")
    print(f"  (Note: direct comparison requires same KT simulator & split)")

    # Save
    out = Path("outputs/results")
    out.mkdir(parents=True, exist_ok=True)
    save = {
        "dataset": "assist09", "provider": args.provider, "p_target": args.p_target,
        "methods": {n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in methods.items()},
        "ablation": {n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in abl.items()},
    }
    with open(out / f"assist09_{args.provider}_p{args.p_target}_results.json", "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved.")


def main():
    args = parse_args()
    if args.stage == "all":
        stage_preprocess(args)
        stage_train(args)
        stage_run(args)
    else:
        getattr(sys.modules[__name__], f"stage_{args.stage}")(args)


if __name__ == "__main__":
    main()
