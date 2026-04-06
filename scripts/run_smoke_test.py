#!/usr/bin/env python3
"""
End-to-end smoke test: full SimPath pipeline on synthetic data.
Validates the entire pipeline without LLM API keys or real datasets.

Usage:
    python scripts/run_smoke_test.py [--n_students 200] [--n_test 50] [--seed 42]
"""

import sys
import os
import argparse
import json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
import numpy as np
from collections import defaultdict

from simpath.utils.seeds import set_global_seed
from simpath.utils.config import load_config
from simpath.data.synthetic import generate_synthetic_dataset
from simpath.personas.definitions import THETA_GRIT, THETA_FRAGILE, PersonaParams, _make_dropout_fn
from simpath.personas.realistic import extract_realistic_params
from simpath.personas.simulation import simulate_all_personas
from simpath.paths.retrieval import retrieve_candidate_pool
from simpath.paths.mock_ordering import mock_generate_paths
from simpath.selection.minimax_regret import (
    composite_scores, minimax_regret_select,
    average_select, maximin_select, hurwicz_select, weighted_minimax_select,
)
from simpath.evaluation.offline_metrics import evaluate_path
from simpath.evaluation.statistical_tests import paired_comparison, ablation_comparison
from simpath.baselines.dkt_random import recommend_random
from simpath.baselines.dkt_rulebased import recommend_rulebased
from simpath.kt.simulate import simulate_kt, delta_mastery


def parse_args():
    p = argparse.ArgumentParser(description="SimPath smoke test")
    p.add_argument("--n_students", type=int, default=200)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="outputs/results")
    return p.parse_args()


def run_main_experiment(test_students, questions, skill_to_questions, cfg):
    """Phase 2: Main experiment — SimPath + baselines on all test students."""
    L = cfg["paths"]["L"]
    K = cfg["paths"]["K"]
    N_sim = cfg["personas"]["N_sim"]
    weights = (
        cfg["selection"]["weights"]["w1"],
        cfg["selection"]["weights"]["w2"],
        cfg["selection"]["weights"]["w3"],
    )

    methods_results = {
        "SimPath (minimax)": defaultdict(list),
        "B1: DKT+Random": defaultdict(list),
        "B2: DKT+RuleBased": defaultdict(list),
        "B6: SimPath-NoRobust": defaultdict(list),
    }
    ablation_strategy = {
        "minimax_regret": defaultdict(list),
        "average": defaultdict(list),
        "maximin": defaultdict(list),
        "hurwicz_0.3": defaultdict(list),
        "hurwicz_0.7": defaultdict(list),
        "weighted_minimax": defaultdict(list),
    }

    for idx, student in enumerate(test_students):
        if idx % 10 == 0:
            print(f"  Student {idx + 1}/{len(test_students)}...")

        mastery = student["mastery"]
        sid = student["student_id"]
        theta_real = extract_realistic_params(student["features"])
        target_kcs = [kc for kc, m in mastery.items() if m < 0.6]
        if not target_kcs:
            target_kcs = list(mastery.keys())

        # --- SimPath pipeline ---
        pool = retrieve_candidate_pool(
            mastery=mastery, question_bank=questions,
            skill_to_questions=skill_to_questions, L=L,
        )
        candidate_paths = mock_generate_paths(mastery, pool, K=K, L=L)

        S = simulate_all_personas(
            mastery=mastery, candidate_paths=candidate_paths,
            theta_real=theta_real, student_id=sid,
            N_sim=N_sim, target_kcs=target_kcs,
        )
        phi = composite_scores(S, weights)

        # Selection strategies
        best_idx, _, _ = minimax_regret_select(phi)
        strategy_map = {
            "minimax_regret": best_idx,
            "average": average_select(phi),
            "maximin": maximin_select(phi),
            "hurwicz_0.3": hurwicz_select(phi, 0.3),
            "hurwicz_0.7": hurwicz_select(phi, 0.7),
            "weighted_minimax": weighted_minimax_select(phi),
        }

        eval_kwargs = dict(
            theta_real=theta_real, target_kcs=target_kcs,
            weights=weights, N_sim=N_sim,
        )

        # Evaluate SimPath
        m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                          student_id=sid, **eval_kwargs)
        for k, v in m.items():
            methods_results["SimPath (minimax)"][k].append(v)

        # Evaluate all selection strategies
        for strat_name, strat_idx in strategy_map.items():
            m = evaluate_path(mastery=mastery, path=candidate_paths[strat_idx],
                              student_id=f"{sid}_{strat_name}", **eval_kwargs)
            for k, v in m.items():
                ablation_strategy[strat_name][k].append(v)

        # B1: Random
        set_global_seed(42 + idx)
        random_path = recommend_random(mastery, skill_to_questions, L=L)
        m = evaluate_path(mastery=mastery, path=random_path,
                          student_id=f"{sid}_b1", **eval_kwargs)
        for k, v in m.items():
            methods_results["B1: DKT+Random"][k].append(v)

        # B2: Rule-based
        rule_path = recommend_rulebased(mastery, skill_to_questions, L=L)
        m = evaluate_path(mastery=mastery, path=rule_path,
                          student_id=f"{sid}_b2", **eval_kwargs)
        for k, v in m.items():
            methods_results["B2: DKT+RuleBased"][k].append(v)

        # B6: NoRobust
        norobust_idx = average_select(phi)
        m = evaluate_path(mastery=mastery, path=candidate_paths[norobust_idx],
                          student_id=f"{sid}_b6", **eval_kwargs)
        for k, v in m.items():
            methods_results["B6: SimPath-NoRobust"][k].append(v)

    return methods_results, ablation_strategy


def run_persona_ablation(test_students, questions, skill_to_questions, cfg):
    """Ablation 1: Number of personas."""
    L, K, N_sim = cfg["paths"]["L"], cfg["paths"]["K"], cfg["personas"]["N_sim"]
    weights = np.array([cfg["selection"]["weights"][f"w{i}"] for i in range(1, 4)])

    THETA_SPRINT = PersonaParams(
        name="sprint", learn_rate=0.20, forget_penalty=0.15,
        dropout_base_fn=_make_dropout_fn(0.05, 1.0),
        difficulty_sensitivity=0.3, skip_threshold=0.5,
        p_correct_boost=-0.05, time_multiplier=0.5,
    )
    THETA_METICULOUS = PersonaParams(
        name="meticulous", learn_rate=0.12, forget_penalty=0.03,
        dropout_base_fn=_make_dropout_fn(0.01, 1.0),
        difficulty_sensitivity=0.1, skip_threshold=None,
        p_correct_boost=0.10, time_multiplier=2.0,
    )

    configs = {
        "1-persona (Real)": lambda tr: [tr],
        "2-persona (G+F)": lambda tr: [THETA_GRIT, THETA_FRAGILE],
        "3-persona (default)": lambda tr: [THETA_GRIT, THETA_FRAGILE, tr],
        "5-persona (+S+M)": lambda tr: [THETA_GRIT, THETA_FRAGILE, tr, THETA_SPRINT, THETA_METICULOUS],
    }

    results = {}
    for config_name, persona_fn in configs.items():
        all_metrics = defaultdict(list)
        for student in test_students:
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = [kc for kc, m in mastery.items() if m < 0.6] or list(mastery.keys())
            personas = persona_fn(theta_real)

            pool = retrieve_candidate_pool(mastery, questions, skill_to_questions, L=L)
            candidate_paths = mock_generate_paths(mastery, pool, K=K, L=L)

            n_paths = len(candidate_paths)
            n_personas = len(personas)
            S = np.zeros((n_paths, n_personas, 3))

            for i in range(n_paths):
                for j, persona in enumerate(personas):
                    mg_r, sr_r, ce_r = [], [], []
                    for sim in range(N_sim):
                        from simpath.utils.seeds import simulation_seed
                        seed = simulation_seed(student["student_id"], i, j, sim)
                        set_global_seed(seed)
                        r = simulate_kt(mastery, candidate_paths[i], persona)
                        mg_r.append(delta_mastery(mastery, r.final_mastery, target_kcs))
                        sr_r.append(1.0 - float(r.dropout))
                        ce_r.append(r.steps_completed / max(r.steps_total, 1))
                    S[i, j, 0], S[i, j, 1], S[i, j, 2] = np.mean(mg_r), np.mean(sr_r), np.mean(ce_r)

            phi = np.einsum('ijk,k->ij', S, weights)
            phi_star = phi.max(axis=0)
            regret = phi_star[None, :] - phi
            best_idx = int(np.argmin(regret.max(axis=1)))

            m = evaluate_path(
                mastery=mastery, path=candidate_paths[best_idx],
                theta_real=theta_real, target_kcs=target_kcs,
                weights=tuple(weights), N_sim=N_sim,
                student_id=student["student_id"],
            )
            for k, v in m.items():
                all_metrics[k].append(v)

        results[config_name] = {k: float(np.mean(v)) for k, v in all_metrics.items()}
    return results


def run_nsim_ablation(test_students, questions, skill_to_questions, cfg):
    """Ablation 5: N_sim variance."""
    L, K = cfg["paths"]["L"], cfg["paths"]["K"]
    weights = tuple(cfg["selection"]["weights"][f"w{i}"] for i in range(1, 4))

    results = {}
    for n_sim in [1, 5, 10, 20, 50]:
        all_metrics = defaultdict(list)
        for student in test_students:
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = [kc for kc, m in mastery.items() if m < 0.6] or list(mastery.keys())

            pool = retrieve_candidate_pool(mastery, questions, skill_to_questions, L=L)
            paths = mock_generate_paths(mastery, pool, K=K, L=L)
            S = simulate_all_personas(mastery, paths, theta_real,
                                      student["student_id"], n_sim, target_kcs)
            phi = composite_scores(S, weights)
            best_idx, _, _ = minimax_regret_select(phi)

            m = evaluate_path(mastery=mastery, path=paths[best_idx],
                              theta_real=theta_real, target_kcs=target_kcs,
                              N_sim=n_sim, student_id=student["student_id"])
            for k, v in m.items():
                all_metrics[k].append(v)

        results[n_sim] = {
            "MG_mean": float(np.mean(all_metrics["MG"])),
            "MG_std": float(np.std(all_metrics["MG"])),
            "SR_std": float(np.std(all_metrics["SR"])),
            "RI_mean": float(np.mean(all_metrics["RI"])),
            "RI_std": float(np.std(all_metrics["RI"])),
        }
    return results


def run_persona_validation(students):
    """Phase 1: Persona validation clustering."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    features_matrix = np.array([
        [s["features"]["avg_session_length"],
         s["features"]["dropout_rates"].get(2, 0.0),
         s["features"]["avg_elapsed_time"],
         s["features"]["skip_rate"],
         s["features"]["overall_accuracy"]]
        for s in students
    ])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_matrix)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    theta_grit_vec = scaler.transform([[15, 0.02, 35, 0.02, 0.7]])
    theta_frag_vec = scaler.transform([[5, 0.30, 15, 0.20, 0.3]])

    cluster_info = []
    for i, centroid in enumerate(kmeans.cluster_centers_):
        c = centroid.reshape(1, -1)
        cluster_info.append({
            "cluster": i,
            "size": int((labels == i).sum()),
            "cos_grit": float(cosine_similarity(c, theta_grit_vec)[0, 0]),
            "cos_frag": float(cosine_similarity(c, theta_frag_vec)[0, 0]),
        })
    return cluster_info


def print_results(methods_results, ablation_strategy, persona_ablation,
                  nsim_ablation, cluster_info):
    """Print all results in formatted tables."""
    print("\n" + "=" * 70)
    print("PERSONA VALIDATION CLUSTERING")
    print("=" * 70)
    for c in cluster_info:
        print(f"  Cluster {c['cluster']} (n={c['size']}): "
              f"cos(Gritty)={c['cos_grit']:.3f}, cos(Fragile)={c['cos_frag']:.3f}")

    print("\n" + "=" * 70)
    print("MAIN RESULTS")
    print("=" * 70)
    print(f"\n{'Method':<25} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 60)
    for name, res in methods_results.items():
        print(f"{name:<25} {np.mean(res['MG']):8.4f} {np.mean(res['SR']):8.4f} "
              f"{np.mean(res['CE']):8.4f} {np.mean(res['RI']):8.4f}")

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (SimPath vs Baselines)")
    print("=" * 70)
    sp = methods_results["SimPath (minimax)"]
    for metric in ["MG", "SR", "CE", "RI"]:
        print(f"\n--- {metric} ---")
        for bname in ["B1: DKT+Random", "B2: DKT+RuleBased", "B6: SimPath-NoRobust"]:
            br = methods_results[bname]
            t = paired_comparison(sp[metric], br[metric], n_comparisons=12)
            sig = "*" if t["significant"] else " "
            print(f"  vs {bname:<22} p={t['p_value']:.4f} d={t['cohens_d']:+.3f} "
                  f"[{t['test']}] {sig}")

    print("\n" + "=" * 70)
    print("ABLATION: Selection Strategy")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 55)
    for sname, sres in ablation_strategy.items():
        print(f"{sname:<20} {np.mean(sres['MG']):8.4f} {np.mean(sres['SR']):8.4f} "
              f"{np.mean(sres['CE']):8.4f} {np.mean(sres['RI']):8.4f}")

    ri_by_strat = {n: r["RI"] for n, r in ablation_strategy.items()}
    kw = ablation_comparison(ri_by_strat)
    print(f"\nKruskal-Wallis (RI): H={kw['kruskal_wallis']['statistic']:.4f}, "
          f"p={kw['kruskal_wallis']['p_value']:.4f}")

    print("\n" + "=" * 70)
    print("ABLATION: Number of Personas")
    print("=" * 70)
    print(f"\n{'Config':<25} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
    print("-" * 60)
    for cname, m in persona_ablation.items():
        print(f"{cname:<25} {m['MG']:8.4f} {m['SR']:8.4f} "
              f"{m['CE']:8.4f} {m['RI']:8.4f}")

    print("\n" + "=" * 70)
    print("ABLATION: Simulation Count (N_sim)")
    print("=" * 70)
    print(f"\n{'N_sim':<10} {'MG_mean':>10} {'MG_std':>10} {'RI_mean':>10} {'RI_std':>10}")
    print("-" * 50)
    for n, v in nsim_ablation.items():
        print(f"{n:<10} {v['MG_mean']:10.4f} {v['MG_std']:10.4f} "
              f"{v['RI_mean']:10.4f} {v['RI_std']:10.4f}")


def main():
    args = parse_args()
    cfg = load_config()
    set_global_seed(args.seed)

    print("=" * 70)
    print("SimPath End-to-End Smoke Test (Synthetic Data)")
    print(f"  Students: {args.n_students}, Test: {args.n_test}, Seed: {args.seed}")
    print("=" * 70)

    # Phase 0: Data
    print("\n[Phase 0] Generating synthetic dataset...")
    students, questions, skill_to_questions = generate_synthetic_dataset(
        n_students=args.n_students, n_questions=100, n_kcs=10, seed=args.seed,
    )
    test_students = students[args.n_students - args.n_test:]
    print(f"  Total: {len(students)}, Test: {len(test_students)}")

    # Phase 1: Persona validation
    print("\n[Phase 1] Persona validation clustering...")
    cluster_info = run_persona_validation(students)

    # Phase 2: Main experiment
    print("\n[Phase 2] Main experiment (SimPath + baselines)...")
    methods_results, ablation_strategy = run_main_experiment(
        test_students, questions, skill_to_questions, cfg,
    )

    # Phase 3: Persona ablation (on subset for speed)
    ablation_subset = test_students[:20]
    print(f"\n[Phase 3] Persona ablation (n={len(ablation_subset)})...")
    persona_ablation = run_persona_ablation(
        ablation_subset, questions, skill_to_questions, cfg,
    )

    # Phase 4: N_sim ablation
    print(f"\n[Phase 4] N_sim ablation (n={len(ablation_subset)})...")
    nsim_ablation = run_nsim_ablation(
        ablation_subset, questions, skill_to_questions, cfg,
    )

    # Print everything
    print_results(methods_results, ablation_strategy, persona_ablation,
                  nsim_ablation, cluster_info)

    # Save JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_data = {
        "main": {n: {k: [float(x) for x in v] for k, v in r.items()}
                 for n, r in methods_results.items()},
        "ablation_strategy": {n: {k: [float(x) for x in v] for k, v in r.items()}
                              for n, r in ablation_strategy.items()},
        "ablation_persona": persona_ablation,
        "ablation_nsim": {str(k): v for k, v in nsim_ablation.items()},
        "cluster_info": cluster_info,
    }
    with open(out_dir / "smoke_test_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_dir / 'smoke_test_results.json'}")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
