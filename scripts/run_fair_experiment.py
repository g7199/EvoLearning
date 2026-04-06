#!/usr/bin/env python3
"""
SimPath Fair Experiment — follows standard LPR evaluation protocol.
EP = (Ee - Es) / (Esup - Es), fixed target_kcs, same pool for all methods.

Usage:
    python scripts/run_fair_experiment.py --provider anthropic --n_test 200
    python scripts/run_fair_experiment.py --provider mock --n_test 5  # quick test
"""

import sys, os, argparse, json, pickle, warnings
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
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
from simpath.evaluation.offline_metrics import evaluate_path, compute_EP
from simpath.evaluation.statistical_tests import paired_comparison
from simpath.baselines.dkt_random import recommend_random
from simpath.baselines.dkt_rulebased import recommend_rulebased


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "both"])
    p.add_argument("--dataset", default="ednet")
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--p_target", type=int, default=8, help="Number of target KCs per student")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def get_fixed_target_kcs(mastery: dict, p: int) -> list:
    """Get p weakest KCs as fixed target set. Same for ALL methods."""
    sorted_kcs = sorted(mastery.items(), key=lambda x: x[1])
    weak = [kc for kc, m in sorted_kcs if m < 0.6]
    if len(weak) >= p:
        return weak[:p]
    return [kc for kc, _ in sorted_kcs[:p]]


def make_path_generator(provider):
    if provider == "mock":
        def gen(mastery, pool, K=5, L=8, **kw):
            return mock_generate_paths(mastery, pool, K, L), \
                {"pool_hit_rate": 1.0, "fill_rate": 0.0, "constraint_violation_rate": 0.0}
        return gen
    else:
        from simpath.llm.client import LLMClient
        from simpath.paths.ordering import generate_candidate_paths_llm
        model = "claude-sonnet-4-6" if provider == "anthropic" else None
        client = LLMClient(provider=provider, model=model)
        print(f"  LLM: {client.provider}/{client.model}")
        def gen(mastery, pool, K=5, L=8, client=client, **kw):
            return generate_candidate_paths_llm(mastery, pool, client, K, L, **kw)
        return gen


def run_experiment(test_students, questions, s2q, cfg, path_gen, provider_name, args):
    L = cfg["paths"]["L"]
    K = cfg["paths"]["K"]
    N_sim = cfg["personas"]["N_sim"]
    weights = (cfg["selection"]["weights"]["w1"],
               cfg["selection"]["weights"]["w2"],
               cfg["selection"]["weights"]["w3"])

    methods = {n: defaultdict(list) for n in
        ["SimPath", "B1:Random", "B2:RuleBased", "B6:NoRobust"]}
    abl = {n: defaultdict(list) for n in
        ["minimax", "average", "maximin", "hurwicz_0.3", "hurwicz_0.7", "weighted"]}
    llm_stats_all = []

    for idx, student in enumerate(test_students):
        if idx % 20 == 0:
            print(f"  [{provider_name}] {idx+1}/{len(test_students)}...")

        mastery = student["mastery"]
        sid = student["student_id"]
        theta_real = extract_realistic_params(student["features"])

        # FIXED target KCs — same for ALL methods (fair!)
        target_kcs = get_fixed_target_kcs(mastery, args.p_target)

        # SAME pool for ALL methods (fair!)
        pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)

        if args.verbose and idx == 0:
            print(f"    target_kcs ({len(target_kcs)}): {target_kcs[:5]}...")
            print(f"    pool size: {len(pool)}")

        eval_kw = dict(theta_real=theta_real, target_kcs=target_kcs,
                        weights=weights, N_sim=N_sim)

        # === SimPath: retrieve + order + simulate + select ===
        candidate_paths, llm_stats = path_gen(mastery, pool, K=K, L=L, seed_base=42+idx)
        llm_stats_all.append(llm_stats)

        S = simulate_all_personas(mastery, candidate_paths, theta_real, sid, N_sim, target_kcs)
        phi = composite_scores(S, weights)
        best_idx, _, _ = minimax_regret_select(phi)

        m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                          student_id=sid, path_idx=best_idx, **eval_kw)
        for k, v in m.items():
            methods["SimPath"][k].append(v)

        # === Selection strategy ablation ===
        for sname, sidx in [("minimax", best_idx), ("average", average_select(phi)),
                            ("maximin", maximin_select(phi)),
                            ("hurwicz_0.3", hurwicz_select(phi, 0.3)),
                            ("hurwicz_0.7", hurwicz_select(phi, 0.7)),
                            ("weighted", weighted_minimax_select(phi))]:
            m = evaluate_path(mastery=mastery, path=candidate_paths[sidx],
                              student_id=sid, path_idx=sidx, **eval_kw)
            for k, v in m.items():
                abl[sname][k].append(v)

        # === B1: Random (SAME pool, SAME target_kcs) ===
        set_global_seed(42 + idx)
        random_path = recommend_random(mastery, s2q, L=L, pool=pool)
        m = evaluate_path(mastery=mastery, path=random_path,
                          student_id=f"{sid}_b1", path_idx=0, **eval_kw)
        for k, v in m.items():
            methods["B1:Random"][k].append(v)

        # === B2: RuleBased (SAME pool, SAME target_kcs) ===
        rule_path = recommend_rulebased(mastery, s2q, L=L, pool=pool)
        m = evaluate_path(mastery=mastery, path=rule_path,
                          student_id=f"{sid}_b2", path_idx=0, **eval_kw)
        for k, v in m.items():
            methods["B2:RuleBased"][k].append(v)

        # === B6: SimPath-NoRobust (average selection, SAME target_kcs) ===
        norobust_idx = average_select(phi)
        m = evaluate_path(mastery=mastery, path=candidate_paths[norobust_idx],
                          student_id=f"{sid}_b6", path_idx=norobust_idx, **eval_kw)
        for k, v in m.items():
            methods["B6:NoRobust"][k].append(v)

    # Aggregate LLM stats
    agg = {}
    for key in ["pool_hit_rate", "fill_rate", "constraint_violation_rate"]:
        vals = [s.get(key, 0) for s in llm_stats_all if s]
        agg[key] = float(np.mean(vals)) if vals else 0

    return methods, abl, agg


def print_results(methods, abl, llm_stats, provider, dataset):
    print(f"\n{'='*70}")
    print(f"RESULTS — {dataset.upper()} / {provider.upper()}")
    if llm_stats:
        print(f"  LLM: pool_hit={llm_stats['pool_hit_rate']:.3f}, "
              f"fill={llm_stats['fill_rate']:.3f}, "
              f"violation={llm_stats['constraint_violation_rate']:.3f}")
    print(f"  Protocol: EP=(Ee-Es)/(1-Es), fixed target_kcs, same pool for all")
    print(f"{'='*70}")

    print(f"\n{'Method':<20} {'EP':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 55)
    for name, res in methods.items():
        print(f"{name:<20} {np.mean(res['EP']):8.4f} {np.mean(res['SR']):8.4f} "
              f"{np.mean(res['CE']):8.4f} {np.mean(res['RI']):8.4f}")

    print(f"\n--- Statistical Tests (Bonferroni α=0.05/12≈0.004) ---")
    sp = methods["SimPath"]
    for metric in ["EP", "SR", "CE", "RI"]:
        print(f"\n  {metric}:")
        for bn in ["B1:Random", "B2:RuleBased", "B6:NoRobust"]:
            t = paired_comparison(sp[metric], methods[bn][metric], n_comparisons=12)
            sig = "***" if t["p_value"] < 0.001 else ("**" if t["p_value"] < 0.01 else ("*" if t["significant"] else ""))
            print(f"    vs {bn:<15} p={t['p_value']:.4f} d={t['cohens_d']:+.3f} {sig}")

    print(f"\n--- Selection Strategy ---")
    print(f"{'Strategy':<15} {'EP':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 50)
    for sn, sr in abl.items():
        print(f"{sn:<15} {np.mean(sr['EP']):8.4f} {np.mean(sr['SR']):8.4f} "
              f"{np.mean(sr['CE']):8.4f} {np.mean(sr['RI']):8.4f}")


def main():
    args = parse_args()
    cfg = load_config()
    set_global_seed(args.seed)

    pkl_path = f"data/processed/{args.dataset}/{args.dataset}_processed.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    students = data["students"]
    questions = data["questions"]
    s2q = data["skill_to_questions"]

    eligible = [s for s in students if len(s["held_out"]) >= 5]
    test_students = eligible[-min(args.n_test, len(eligible)):]
    print(f"Dataset={args.dataset}, Test={len(test_students)}, p_target={args.p_target}")

    providers = []
    if args.provider == "both":
        providers = ["anthropic", "openai"]
    else:
        providers = [args.provider]

    all_results = {}
    for prov in providers:
        print(f"\n{'='*60}\nProvider: {prov}\n{'='*60}")
        path_gen = make_path_generator(prov)
        methods, abl, llm_stats = run_experiment(
            test_students, questions, s2q, cfg, path_gen, prov, args)
        print_results(methods, abl, llm_stats, prov, args.dataset)
        all_results[prov] = {
            "methods": {n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in methods.items()},
            "ablation": {n: {k: [float(x) for x in v] for k, v in r.items()} for n, r in abl.items()},
            "llm_stats": llm_stats,
        }

    # Save
    out = Path("outputs/results")
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"{args.dataset}_{args.provider}_fair_results.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": args.dataset, "protocol": "EP_fixed_target_same_pool", **all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
