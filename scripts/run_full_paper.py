#!/usr/bin/env python3
"""
FINAL paper experiment — everything in one script.
1. Main experiment: 2 datasets × 5 seeds × 200 students × 6 methods
2. Ablation studies: persona count, K sweep, N_sim sweep, selection strategy
All with LLM (Anthropic Claude Sonnet 4.6).
"""

import sys, os, argparse, json, pickle, warnings, random, time
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from simpath.utils.seeds import set_global_seed
from simpath.utils.config import load_config
from simpath.personas.definitions import THETA_GRIT, THETA_FRAGILE, PersonaParams, _make_dropout_fn
from simpath.personas.realistic import extract_realistic_params
from simpath.personas.simulation import simulate_all_personas
from simpath.paths.retrieval import retrieve_candidate_pool
from simpath.paths.mock_ordering import mock_generate_paths
from simpath.paths.ordering import generate_candidate_paths_llm
from simpath.selection.minimax_regret import (
    composite_scores, minimax_regret_select,
    average_select, maximin_select, hurwicz_select, weighted_minimax_select,
)
from simpath.evaluation.offline_metrics import evaluate_path, compute_EP
from simpath.evaluation.statistical_tests import paired_comparison
from simpath.baselines.dkt_random import recommend_random
from simpath.baselines.dkt_rulebased import recommend_rulebased
from simpath.baselines.rl_dkt import train_rl_dkt, recommend_rl_dkt
from simpath.baselines.lprekl import recommend_lprekl
from simpath.llm.client import LLMClient
from simpath.kt.simulate import simulate_kt, delta_mastery

SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ["assist09", "junyi"]


def get_target_kcs(mastery, p=5):
    sorted_kcs = sorted(mastery.items(), key=lambda x: x[1])
    weak = [kc for kc, m in sorted_kcs if m < 0.6]
    return weak[:p] if len(weak) >= p else [kc for kc, _ in sorted_kcs[:p]]


def run_main_experiment(dataset_name, n_test=200, n_seeds=5, p_target=5):
    """Main experiment: 5 seeds × 200 students × 6 methods."""
    print(f"\n{'#'*70}")
    print(f"# MAIN EXPERIMENT: {dataset_name.upper()}")
    print(f"# {n_seeds} seeds × {n_test} students × 6 methods")
    print(f"{'#'*70}")

    cfg = load_config()
    with open(f"data/processed/{dataset_name}/{dataset_name}_processed.pkl", "rb") as f:
        data = pickle.load(f)

    students = data["students"]
    questions = data["questions"]
    s2q = data["skill_to_questions"]
    kc_list = data["kc_list"]

    eligible = [s for s in students if len(s["held_out"]) >= 5]
    test_students = eligible[-min(n_test, len(eligible)):]
    print(f"  Students={len(students)}, Test={len(test_students)}, KC={len(kc_list)}, Q={len(questions)}")

    L, K, N_sim = cfg["paths"]["L"], cfg["paths"]["K"], cfg["personas"]["N_sim"]
    weights = (cfg["selection"]["weights"]["w1"],
               cfg["selection"]["weights"]["w2"],
               cfg["selection"]["weights"]["w3"])

    # LLM clients
    simpath_client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
    lprekl_client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
    print(f"  LLM: {simpath_client.model}")

    # Train RL-DKT
    print(f"\n  Training RL-DKT (10K episodes)...")
    common_pool = questions[:min(500, len(questions))]
    rl_agent = train_rl_dkt(students[:500], common_pool, kc_list,
                             n_episodes=10000, L=L, target_p=p_target)

    # Run per seed
    all_seed_results = []
    for si, seed in enumerate(SEEDS[:n_seeds]):
        t0 = time.time()
        set_global_seed(seed)
        print(f"\n  === Seed {seed} ({si+1}/{n_seeds}) ===")

        methods = {n: defaultdict(list) for n in
            ["SimPath", "B1:Random", "B2:RuleBased", "B3:RL-DKT", "B4:LPReKL", "B6:NoRobust"]}

        for idx, student in enumerate(test_students):
            if idx % 50 == 0:
                print(f"    Student {idx+1}/{len(test_students)}...")

            mastery = student["mastery"]
            sid = f"{student['student_id']}_s{seed}"
            theta_real = extract_realistic_params(student["features"])
            target_kcs = get_target_kcs(mastery, p_target)

            pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
            eval_kw = dict(theta_real=theta_real, target_kcs=target_kcs,
                            weights=weights, N_sim=N_sim)

            # SimPath (LLM scoring)
            candidate_paths, _ = generate_candidate_paths_llm(
                mastery, pool, simpath_client, K=K, L=L, seed_base=seed+idx)
            S = simulate_all_personas(mastery, candidate_paths, theta_real,
                                       sid, N_sim, target_kcs)
            phi = composite_scores(S, weights)
            best_idx, _, _ = minimax_regret_select(phi)

            m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                              student_id=sid, path_idx=best_idx, **eval_kw)
            for k, v in m.items():
                methods["SimPath"][k].append(v)

            # Baselines (same pool)
            set_global_seed(seed + idx)
            for bn, bp in [
                ("B1:Random", recommend_random(mastery, s2q, L=L, pool=pool)),
                ("B2:RuleBased", recommend_rulebased(mastery, s2q, L=L, pool=pool)),
                ("B3:RL-DKT", recommend_rl_dkt(rl_agent, mastery, common_pool, kc_list, L=L)),
                ("B4:LPReKL", recommend_lprekl(mastery, pool, kc_list, L=L, llm_client=lprekl_client)),
            ]:
                m = evaluate_path(mastery=mastery, path=bp,
                                  student_id=f"{sid}_{bn}", path_idx=0, **eval_kw)
                for k, v in m.items():
                    methods[bn][k].append(v)

            # B6: NoRobust
            norobust_idx = average_select(phi)
            m = evaluate_path(mastery=mastery, path=candidate_paths[norobust_idx],
                              student_id=f"{sid}_b6", path_idx=norobust_idx, **eval_kw)
            for k, v in m.items():
                methods["B6:NoRobust"][k].append(v)

        seed_result = {n: {k: float(np.mean(v)) for k, v in r.items()} for n, r in methods.items()}
        all_seed_results.append(seed_result)
        elapsed = time.time() - t0
        print(f"    Seed {seed} done in {elapsed/60:.1f} min")
        for name, metrics in seed_result.items():
            print(f"      {name:<15} EP={metrics['EP']:.4f} SR={metrics['SR']:.4f} CE={metrics['CE']:.4f} RI={metrics['RI']:.4f}")

    # Aggregate
    method_names = list(all_seed_results[0].keys())
    final = {}
    print(f"\n  {'='*60}")
    print(f"  FINAL — {dataset_name.upper()} ({n_seeds} seeds)")
    print(f"  {'='*60}")
    print(f"  {'Method':<15} {'EP':>12} {'SR':>12} {'CE':>12} {'RI':>12}")
    print(f"  {'-'*63}")
    for name in method_names:
        vals = {m: [r[name][m] for r in all_seed_results] for m in ["EP","SR","CE","RI"]}
        means = {m: np.mean(vals[m]) for m in vals}
        stds = {m: np.std(vals[m]) for m in vals}
        final[name] = {"means": means, "stds": stds}
        print(f"  {name:<15} {means['EP']:.4f}±{stds['EP']:.4f} {means['SR']:.4f}±{stds['SR']:.4f} "
              f"{means['CE']:.4f}±{stds['CE']:.4f} {means['RI']:.4f}±{stds['RI']:.4f}")

    return final


def run_ablations(dataset_name="assist09", n_test=100, seed=42):
    """Ablation studies on ASSIST09."""
    print(f"\n{'#'*70}")
    print(f"# ABLATION STUDIES: {dataset_name.upper()}")
    print(f"{'#'*70}")

    cfg = load_config()
    set_global_seed(seed)

    with open(f"data/processed/{dataset_name}/{dataset_name}_processed.pkl", "rb") as f:
        data = pickle.load(f)

    students = data["students"]
    questions = data["questions"]
    s2q = data["skill_to_questions"]
    kc_list = data["kc_list"]

    eligible = [s for s in students if len(s["held_out"]) >= 5]
    test_students = eligible[-min(n_test, len(eligible)):]

    L = cfg["paths"]["L"]
    weights_tuple = (cfg["selection"]["weights"]["w1"],
                     cfg["selection"]["weights"]["w2"],
                     cfg["selection"]["weights"]["w3"])
    weights_arr = np.array(weights_tuple)

    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6")
    ablation_results = {}

    # ─── Ablation 1: Persona Count ───
    print(f"\n  --- Ablation 1: Persona Count ---")
    THETA_SPRINT = PersonaParams(
        name="sprint", learn_rate=0.20, forget_penalty=0.15,
        dropout_base_fn=_make_dropout_fn(0.05, 1.0),
        difficulty_sensitivity=0.3, skip_threshold=0.5,
        p_correct_boost=-0.05, time_multiplier=0.5)
    THETA_METICULOUS = PersonaParams(
        name="meticulous", learn_rate=0.12, forget_penalty=0.03,
        dropout_base_fn=_make_dropout_fn(0.01, 1.0),
        difficulty_sensitivity=0.1, skip_threshold=None,
        p_correct_boost=0.10, time_multiplier=2.0)

    persona_configs = {
        "1(Real)": lambda tr: [tr],
        "2(G+F)": lambda tr: [THETA_GRIT, THETA_FRAGILE],
        "3(default)": lambda tr: [THETA_GRIT, THETA_FRAGILE, tr],
        "5(+S+M)": lambda tr: [THETA_GRIT, THETA_FRAGILE, tr, THETA_SPRINT, THETA_METICULOUS],
    }

    abl1 = {}
    for config_name, persona_fn in persona_configs.items():
        metrics_all = defaultdict(list)
        for idx, student in enumerate(test_students):
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = get_target_kcs(mastery, 5)
            personas = persona_fn(theta_real)

            pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
            paths, _ = generate_candidate_paths_llm(mastery, pool, client, K=5, L=L, seed_base=seed+idx)

            n_p, n_j = len(paths), len(personas)
            S = np.zeros((n_p, n_j, 3))
            for i in range(n_p):
                for j, pers in enumerate(personas):
                    ep_r, sr_r, ce_r = [], [], []
                    for sim in range(10):
                        from simpath.utils.seeds import simulation_seed
                        set_global_seed(simulation_seed(student["student_id"], i, j, sim))
                        r = simulate_kt(mastery, paths[i], pers)
                        ep_r.append(compute_EP(mastery, r.final_mastery, target_kcs))
                        sr_r.append(1.0 - float(r.dropout))
                        ce_r.append(r.steps_completed / max(r.steps_total, 1))
                    S[i, j, 0], S[i, j, 1], S[i, j, 2] = np.mean(ep_r), np.mean(sr_r), np.mean(ce_r)

            phi = np.einsum('ijk,k->ij', S, weights_arr)
            best = int(np.argmin((phi.max(0)[None, :] - phi).max(1)))

            m = evaluate_path(mastery=mastery, path=paths[best], theta_real=theta_real,
                              target_kcs=target_kcs, weights=weights_tuple, N_sim=10,
                              student_id=student["student_id"], path_idx=best)
            for k, v in m.items():
                metrics_all[k].append(v)

        abl1[config_name] = {k: float(np.mean(v)) for k, v in metrics_all.items()}
        print(f"    {config_name:<12} EP={abl1[config_name]['EP']:.4f} SR={abl1[config_name]['SR']:.4f} "
              f"CE={abl1[config_name]['CE']:.4f} RI={abl1[config_name]['RI']:.4f}")
    ablation_results["persona_count"] = abl1

    # ─── Ablation 2: Selection Strategy ───
    print(f"\n  --- Ablation 2: Selection Strategy ---")
    abl2 = {n: defaultdict(list) for n in
        ["minimax", "average", "maximin", "hurwicz_0.3", "hurwicz_0.7", "weighted"]}

    for idx, student in enumerate(test_students):
        mastery = student["mastery"]
        theta_real = extract_realistic_params(student["features"])
        target_kcs = get_target_kcs(mastery, 5)

        pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
        paths, _ = generate_candidate_paths_llm(mastery, pool, client, K=5, L=L, seed_base=seed+idx)
        S = simulate_all_personas(mastery, paths, theta_real,
                                   student["student_id"], 10, target_kcs)
        phi = composite_scores(S, weights_tuple)

        for sname, sidx in [("minimax", minimax_regret_select(phi)[0]),
                            ("average", average_select(phi)),
                            ("maximin", maximin_select(phi)),
                            ("hurwicz_0.3", hurwicz_select(phi, 0.3)),
                            ("hurwicz_0.7", hurwicz_select(phi, 0.7)),
                            ("weighted", weighted_minimax_select(phi))]:
            m = evaluate_path(mastery=mastery, path=paths[sidx], theta_real=theta_real,
                              target_kcs=target_kcs, weights=weights_tuple, N_sim=10,
                              student_id=f"{student['student_id']}_{sname}", path_idx=sidx)
            for k, v in m.items():
                abl2[sname][k].append(v)

    for sn in abl2:
        abl2[sn] = {k: float(np.mean(v)) for k, v in abl2[sn].items()}
        print(f"    {sn:<12} EP={abl2[sn]['EP']:.4f} SR={abl2[sn]['SR']:.4f} "
              f"CE={abl2[sn]['CE']:.4f} RI={abl2[sn]['RI']:.4f}")
    ablation_results["selection_strategy"] = abl2

    # ─── Ablation 3: N_sim ───
    print(f"\n  --- Ablation 3: N_sim ---")
    abl3 = {}
    for n_sim in [1, 5, 10, 20]:
        metrics_all = defaultdict(list)
        for idx, student in enumerate(test_students[:50]):  # smaller subset for speed
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = get_target_kcs(mastery, 5)
            pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
            paths, _ = generate_candidate_paths_llm(mastery, pool, client, K=5, L=L, seed_base=seed+idx)
            S = simulate_all_personas(mastery, paths, theta_real,
                                       student["student_id"], n_sim, target_kcs)
            phi = composite_scores(S, weights_tuple)
            best, _, _ = minimax_regret_select(phi)
            m = evaluate_path(mastery=mastery, path=paths[best], theta_real=theta_real,
                              target_kcs=target_kcs, N_sim=n_sim,
                              student_id=student["student_id"], path_idx=best)
            for k, v in m.items():
                metrics_all[k].append(v)
        abl3[n_sim] = {k: f"{np.mean(v):.4f}±{np.std(v):.4f}" for k, v in metrics_all.items()}
        print(f"    N_sim={n_sim:<3} EP={np.mean(metrics_all['EP']):.4f}±{np.std(metrics_all['EP']):.4f}")
    ablation_results["n_sim"] = abl3

    # ─── Ablation 4: K (candidate paths) ───
    print(f"\n  --- Ablation 4: K (candidate paths) ---")
    abl4 = {}
    for K_val in [3, 5, 8]:
        metrics_all = defaultdict(list)
        for idx, student in enumerate(test_students[:50]):
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = get_target_kcs(mastery, 5)
            pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
            paths, _ = generate_candidate_paths_llm(mastery, pool, client, K=K_val, L=L, seed_base=seed+idx)
            S = simulate_all_personas(mastery, paths, theta_real,
                                       student["student_id"], 10, target_kcs)
            phi = composite_scores(S, weights_tuple)
            best, _, _ = minimax_regret_select(phi)
            m = evaluate_path(mastery=mastery, path=paths[best], theta_real=theta_real,
                              target_kcs=target_kcs, N_sim=10,
                              student_id=student["student_id"], path_idx=best)
            for k, v in m.items():
                metrics_all[k].append(v)
        abl4[K_val] = {k: f"{np.mean(v):.4f}±{np.std(v):.4f}" for k, v in metrics_all.items()}
        print(f"    K={K_val:<3} EP={np.mean(metrics_all['EP']):.4f}±{np.std(metrics_all['EP']):.4f}")
    ablation_results["K_paths"] = abl4

    return ablation_results


def main():
    t_start = time.time()

    # ═══ Part 1: Main experiments ═══
    all_main = {}
    for ds in DATASETS:
        pkl = f"data/processed/{ds}/{ds}_processed.pkl"
        if not os.path.exists(pkl):
            print(f"  [SKIP] {ds} not preprocessed")
            continue
        all_main[ds] = run_main_experiment(ds, n_test=200, n_seeds=5, p_target=5)

    # ═══ Part 2: Ablation studies (on ASSIST09) ═══
    ablations = run_ablations("assist09", n_test=100, seed=42)

    # ═══ Save everything ═══
    out = Path("outputs/results")
    out.mkdir(parents=True, exist_ok=True)

    save = {"main": {}, "ablations": ablations}
    for ds, final in all_main.items():
        save["main"][ds] = {
            name: {
                "EP": f"{d['means']['EP']:.4f}±{d['stds']['EP']:.4f}",
                "SR": f"{d['means']['SR']:.4f}±{d['stds']['SR']:.4f}",
                "CE": f"{d['means']['CE']:.4f}±{d['stds']['CE']:.4f}",
                "RI": f"{d['means']['RI']:.4f}±{d['stds']['RI']:.4f}",
            } for name, d in final.items()
        }

    with open(out / "paper_final_results.json", "w") as f:
        json.dump(save, f, indent=2)

    elapsed = (time.time() - t_start) / 3600
    print(f"\n{'='*70}")
    print(f"ALL DONE in {elapsed:.1f} hours")
    print(f"Results: outputs/results/paper_final_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
