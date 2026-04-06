#!/usr/bin/env python3
"""
SimPath Full Experiment Pipeline.

Supports both mock (no API key) and real LLM (OpenAI/Anthropic) modes.

Usage:
    # Mock mode (no API key, heuristic ordering):
    python scripts/run_experiment.py --provider mock

    # OpenAI:
    python scripts/run_experiment.py --provider openai --api_key sk-...
    # or: export OPENAI_API_KEY=sk-... && python scripts/run_experiment.py --provider openai

    # Anthropic:
    python scripts/run_experiment.py --provider anthropic --api_key sk-ant-...
    # or: export ANTHROPIC_API_KEY=sk-ant-... && python scripts/run_experiment.py --provider anthropic

    # Both (for paper — runs GPT-5.4 and Claude Sonnet 4.6):
    python scripts/run_experiment.py --provider both \\
        --openai_key sk-... --anthropic_key sk-ant-...

Options:
    --n_students    Total synthetic students (default: 200)
    --n_test        Test students (default: 50)
    --seed          Random seed (default: 42)
    --model         Override model name (e.g., gpt-4o, claude-sonnet-4-6)
    --provider      mock | openai | anthropic | both
"""

import sys, os, argparse, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
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
    p = argparse.ArgumentParser()
    p.add_argument("--provider", type=str, default="mock",
                   choices=["mock", "openai", "anthropic", "both"])
    p.add_argument("--api_key", type=str, default=None, help="API key (for single provider)")
    p.add_argument("--openai_key", type=str, default=None)
    p.add_argument("--anthropic_key", type=str, default=None)
    p.add_argument("--model", type=str, default=None, help="Override model name")
    p.add_argument("--n_students", type=int, default=200)
    p.add_argument("--n_test", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Quick sanity check: 5 students, N_sim=3, skip ablations")
    p.add_argument("--no-ablation", action="store_true", dest="no_ablation",
                   help="Skip ablation studies (main experiment only)")
    p.add_argument("--output_dir", type=str, default="outputs/results")
    return p.parse_args()


def make_llm_path_generator(provider, api_key=None, model=None):
    """Create a path generation function using LLM."""
    from simpath.llm.client import LLMClient
    from simpath.paths.ordering import generate_candidate_paths_llm

    client = LLMClient(provider=provider, model=model, api_key=api_key)
    print(f"  LLM: {client.provider} / {client.model}")

    def generate(mastery, pool, K=5, L=8, seed_base=42):
        paths, stats = generate_candidate_paths_llm(
            mastery, pool, client, K=K, L=L, seed_base=seed_base,
        )
        return paths, stats

    return generate


def make_mock_path_generator():
    """Create a path generation function using mock heuristics."""
    def generate(mastery, pool, K=5, L=8, seed_base=42):
        paths = mock_generate_paths(mastery, pool, K=K, L=L)
        stats = {"pool_hit_rate": 1.0, "fill_rate": 0.0, "constraint_violation_rate": 0.0}
        return paths, stats
    return generate


def run_pipeline(test_students, questions, skill_to_questions, cfg,
                 path_generator, label="SimPath"):
    """Run full SimPath pipeline with given path generator."""
    L, K, N_sim = cfg["paths"]["L"], cfg["paths"]["K"], cfg["personas"]["N_sim"]
    weights = (cfg["selection"]["weights"]["w1"],
               cfg["selection"]["weights"]["w2"],
               cfg["selection"]["weights"]["w3"])

    results = {
        "SimPath (minimax)": defaultdict(list),
        "B1: DKT+Random": defaultdict(list),
        "B2: DKT+RuleBased": defaultdict(list),
        "B6: SimPath-NoRobust": defaultdict(list),
    }
    ablation_strategy = {
        name: defaultdict(list) for name in
        ["minimax_regret", "average", "maximin", "hurwicz_0.3", "hurwicz_0.7", "weighted_minimax"]
    }
    all_llm_stats = []

    for idx, student in enumerate(test_students):
        if idx % 10 == 0:
            print(f"  [{label}] Student {idx+1}/{len(test_students)}...")

        mastery = student["mastery"]
        sid = student["student_id"]
        theta_real = extract_realistic_params(student["features"])
        target_kcs = [kc for kc, m in mastery.items() if m < 0.6] or list(mastery.keys())
        eval_kw = dict(theta_real=theta_real, target_kcs=target_kcs,
                       weights=weights, N_sim=N_sim)

        # Retrieve + Order
        pool = retrieve_candidate_pool(mastery, questions, skill_to_questions, L=L)
        candidate_paths, llm_stats = path_generator(mastery, pool, K=K, L=L, seed_base=42+idx)
        all_llm_stats.append(llm_stats)

        # Simulate + Select
        S = simulate_all_personas(mastery, candidate_paths, theta_real, sid, N_sim, target_kcs)
        phi = composite_scores(S, weights)
        best_idx, _, _ = minimax_regret_select(phi)

        # Evaluate SimPath
        m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                          student_id=sid, **eval_kw)
        for k, v in m.items():
            results["SimPath (minimax)"][k].append(v)

        # Ablation: selection strategies
        for sname, sidx in [
            ("minimax_regret", best_idx),
            ("average", average_select(phi)),
            ("maximin", maximin_select(phi)),
            ("hurwicz_0.3", hurwicz_select(phi, 0.3)),
            ("hurwicz_0.7", hurwicz_select(phi, 0.7)),
            ("weighted_minimax", weighted_minimax_select(phi)),
        ]:
            m = evaluate_path(mastery=mastery, path=candidate_paths[sidx],
                              student_id=f"{sid}_{sname}", **eval_kw)
            for k, v in m.items():
                ablation_strategy[sname][k].append(v)

        # Baselines
        set_global_seed(42 + idx)
        for bname, bpath in [
            ("B1: DKT+Random", recommend_random(mastery, skill_to_questions, L=L)),
            ("B2: DKT+RuleBased", recommend_rulebased(mastery, skill_to_questions, L=L)),
        ]:
            m = evaluate_path(mastery=mastery, path=bpath,
                              student_id=f"{sid}_{bname}", **eval_kw)
            for k, v in m.items():
                results[bname][k].append(v)

        # B6: NoRobust
        norobust_idx = average_select(phi)
        m = evaluate_path(mastery=mastery, path=candidate_paths[norobust_idx],
                          student_id=f"{sid}_b6", **eval_kw)
        for k, v in m.items():
            results["B6: SimPath-NoRobust"][k].append(v)

    # Aggregate LLM stats
    agg_stats = {}
    if all_llm_stats and "pool_hit_rate" in all_llm_stats[0]:
        for key in ["pool_hit_rate", "fill_rate", "constraint_violation_rate"]:
            vals = [s[key] for s in all_llm_stats if key in s]
            agg_stats[key] = float(np.mean(vals)) if vals else None

    return results, ablation_strategy, agg_stats


def run_persona_ablation(test_students, questions, skill_to_questions, cfg, path_generator):
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

    out = {}
    for cname, persona_fn in configs.items():
        all_m = defaultdict(list)
        for student in test_students:
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = [kc for kc, m in mastery.items() if m < 0.6] or list(mastery.keys())
            personas = persona_fn(theta_real)

            pool = retrieve_candidate_pool(mastery, questions, skill_to_questions, L=L)
            candidate_paths, _ = path_generator(mastery, pool, K=K, L=L)

            n_p, n_j = len(candidate_paths), len(personas)
            S = np.zeros((n_p, n_j, 3))
            for i in range(n_p):
                for j, pers in enumerate(personas):
                    mg_r, sr_r, ce_r = [], [], []
                    for sim in range(N_sim):
                        from simpath.utils.seeds import simulation_seed
                        set_global_seed(simulation_seed(student["student_id"], i, j, sim))
                        r = simulate_kt(mastery, candidate_paths[i], pers)
                        mg_r.append(delta_mastery(mastery, r.final_mastery, target_kcs))
                        sr_r.append(1.0 - float(r.dropout))
                        ce_r.append(r.steps_completed / max(r.steps_total, 1))
                    S[i, j, 0], S[i, j, 1], S[i, j, 2] = np.mean(mg_r), np.mean(sr_r), np.mean(ce_r)

            phi = np.einsum('ijk,k->ij', S, weights)
            best_idx = int(np.argmin((phi.max(0)[None, :] - phi).max(1)))
            m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                              theta_real=theta_real, target_kcs=target_kcs,
                              weights=tuple(weights), N_sim=N_sim,
                              student_id=student["student_id"])
            for k, v in m.items():
                all_m[k].append(v)
        out[cname] = {k: float(np.mean(v)) for k, v in all_m.items()}
    return out


def run_nsim_ablation(test_students, questions, skill_to_questions, cfg, path_generator):
    """Ablation 5: N_sim variance."""
    L, K = cfg["paths"]["L"], cfg["paths"]["K"]
    weights = tuple(cfg["selection"]["weights"][f"w{i}"] for i in range(1, 4))
    out = {}
    for n_sim in [1, 5, 10, 20, 50]:
        all_m = defaultdict(list)
        for student in test_students:
            mastery = student["mastery"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = [kc for kc, m in mastery.items() if m < 0.6] or list(mastery.keys())
            pool = retrieve_candidate_pool(mastery, questions, skill_to_questions, L=L)
            paths, _ = path_generator(mastery, pool, K=K, L=L)
            S = simulate_all_personas(mastery, paths, theta_real,
                                      student["student_id"], n_sim, target_kcs)
            phi = composite_scores(S, weights)
            best, _, _ = minimax_regret_select(phi)
            m = evaluate_path(mastery=mastery, path=paths[best], theta_real=theta_real,
                              target_kcs=target_kcs, N_sim=n_sim,
                              student_id=student["student_id"])
            for k, v in m.items():
                all_m[k].append(v)
        out[n_sim] = {
            "MG_mean": float(np.mean(all_m["MG"])), "MG_std": float(np.std(all_m["MG"])),
            "SR_std": float(np.std(all_m["SR"])),
            "RI_mean": float(np.mean(all_m["RI"])), "RI_std": float(np.std(all_m["RI"])),
        }
    return out


def persona_validation(students):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    feats = np.array([[s["features"]["avg_session_length"],
                        s["features"]["dropout_rates"].get(2, 0.0),
                        s["features"]["avg_elapsed_time"],
                        s["features"]["skip_rate"],
                        s["features"]["overall_accuracy"]] for s in students])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)

    grit_vec = scaler.transform([[15, 0.02, 35, 0.02, 0.7]])
    frag_vec = scaler.transform([[5, 0.30, 15, 0.20, 0.3]])

    info = []
    for i, c in enumerate(km.cluster_centers_):
        c2d = c.reshape(1, -1)
        info.append({
            "cluster": i, "size": int((km.labels_ == i).sum()),
            "cos_grit": float(cosine_similarity(c2d, grit_vec)[0, 0]),
            "cos_frag": float(cosine_similarity(c2d, frag_vec)[0, 0]),
        })
    return info


def print_all(results_by_provider, ablation_strat, persona_abl, nsim_abl,
              cluster_info, llm_stats_by_provider):
    print("\n" + "=" * 70)
    print("PERSONA VALIDATION CLUSTERING")
    print("=" * 70)
    for c in cluster_info:
        print(f"  Cluster {c['cluster']} (n={c['size']}): "
              f"cos(Gritty)={c['cos_grit']:.3f}, cos(Fragile)={c['cos_frag']:.3f}")

    for prov, results in results_by_provider.items():
        print(f"\n{'=' * 70}")
        print(f"MAIN RESULTS — {prov.upper()}")
        if prov in llm_stats_by_provider and llm_stats_by_provider[prov]:
            s = llm_stats_by_provider[prov]
            print(f"  LLM Stats: pool_hit={s.get('pool_hit_rate','N/A'):.3f}, "
                  f"fill={s.get('fill_rate','N/A'):.3f}, "
                  f"violation={s.get('constraint_violation_rate','N/A'):.3f}")
        print("=" * 70)
        print(f"\n{'Method':<25} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
        print("-" * 60)
        for name, res in results.items():
            print(f"{name:<25} {np.mean(res['MG']):8.4f} {np.mean(res['SR']):8.4f} "
                  f"{np.mean(res['CE']):8.4f} {np.mean(res['RI']):8.4f}")

        print(f"\n--- Statistical Tests ---")
        sp = results["SimPath (minimax)"]
        for metric in ["MG", "SR", "CE", "RI"]:
            print(f"\n  {metric}:")
            for bn in ["B1: DKT+Random", "B2: DKT+RuleBased", "B6: SimPath-NoRobust"]:
                t = paired_comparison(sp[metric], results[bn][metric], n_comparisons=12)
                sig = "*" if t["significant"] else " "
                print(f"    vs {bn:<22} p={t['p_value']:.4f} d={t['cohens_d']:+.3f} {sig}")

    # Ablation: strategies (last provider only)
    if ablation_strat:
        print(f"\n{'=' * 70}")
        print("ABLATION: Selection Strategy")
        print("=" * 70)
        print(f"\n{'Strategy':<20} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
        print("-" * 55)
        last_strat = list(ablation_strat.values())[-1]
        for sn, sr in last_strat.items():
            print(f"{sn:<20} {np.mean(sr['MG']):8.4f} {np.mean(sr['SR']):8.4f} "
                  f"{np.mean(sr['CE']):8.4f} {np.mean(sr['RI']):8.4f}")

    if persona_abl:
        print(f"\n{'=' * 70}")
        print("ABLATION: Number of Personas")
        print("=" * 70)
        print(f"\n{'Config':<25} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
        print("-" * 60)
        for cn, m in persona_abl.items():
            print(f"{cn:<25} {m['MG']:8.4f} {m['SR']:8.4f} {m['CE']:8.4f} {m['RI']:8.4f}")

    if nsim_abl:
        print(f"\n{'=' * 70}")
        print("ABLATION: Simulation Count (N_sim)")
        print("=" * 70)
        print(f"\n{'N_sim':<10} {'MG_mean':>10} {'MG_std':>10} {'RI_mean':>10} {'RI_std':>10}")
        print("-" * 50)
        for n, v in nsim_abl.items():
            print(f"{n:<10} {v['MG_mean']:10.4f} {v['MG_std']:10.4f} "
                  f"{v['RI_mean']:10.4f} {v['RI_std']:10.4f}")


def main():
    args = parse_args()
    cfg = load_config()
    set_global_seed(args.seed)

    # --quick: 소량 데이터로 빠르게 파이프라인 검증
    if args.quick:
        args.n_students = 20
        args.n_test = 5
        cfg["personas"]["N_sim"] = 3
        cfg["paths"]["K"] = 3
        print("=" * 70)
        print(f"SimPath QUICK CHECK — provider={args.provider}")
        print(f"  (students=20, test=5, N_sim=3, K=3, ablations 스킵)")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"SimPath Experiment — provider={args.provider}")
        print("=" * 70)

    # Data
    print("\n[Data] Generating synthetic dataset...")
    students, questions, s2q = generate_synthetic_dataset(
        n_students=args.n_students, n_questions=100, n_kcs=10, seed=args.seed)
    test_students = students[args.n_students - args.n_test:]
    ablation_subset = test_students[:min(20, len(test_students))]

    # Persona validation
    print("[Clustering] Persona validation...")
    cluster_info = persona_validation(students)

    # Build provider list
    providers = []
    if args.provider == "mock":
        providers = [("mock", None, None)]
    elif args.provider == "openai":
        providers = [("openai", args.api_key or args.openai_key, args.model)]
    elif args.provider == "anthropic":
        providers = [("anthropic", args.api_key or args.anthropic_key, args.model)]
    elif args.provider == "both":
        providers = [
            ("openai", args.openai_key, args.model or "gpt-5.4-2026-03-05"),
            ("anthropic", args.anthropic_key, args.model or "claude-sonnet-4-6"),
        ]

    results_by_provider = {}
    ablation_strat_by_provider = {}
    llm_stats_by_provider = {}

    for prov, key, model in providers:
        print(f"\n[Pipeline] Running with provider={prov}...")
        if prov == "mock":
            gen = make_mock_path_generator()
        else:
            gen = make_llm_path_generator(prov, api_key=key, model=model)

        res, abl_strat, llm_stats = run_pipeline(
            test_students, questions, s2q, cfg, gen, label=prov)
        results_by_provider[prov] = res
        ablation_strat_by_provider[prov] = abl_strat
        llm_stats_by_provider[prov] = llm_stats

    # Ablations (skip in quick / no-ablation mode)
    skip_ablation = args.quick or args.no_ablation
    persona_abl = {}
    nsim_abl = {}
    if not skip_ablation:
        prov0 = providers[0]
        gen0 = make_mock_path_generator() if prov0[0] == "mock" else make_llm_path_generator(*prov0)

        print(f"\n[Ablation] Persona count (n={len(ablation_subset)})...")
        persona_abl = run_persona_ablation(ablation_subset, questions, s2q, cfg, gen0)

        print(f"[Ablation] N_sim (n={len(ablation_subset)})...")
        nsim_abl = run_nsim_ablation(ablation_subset, questions, s2q, cfg, gen0)
    else:
        print("\n[Skipped] Ablations (use without --quick/--no-ablation to run)")

    # Print
    print_all(results_by_provider, ablation_strat_by_provider, persona_abl,
              nsim_abl, cluster_info, llm_stats_by_provider)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {
        "providers": [p[0] for p in providers],
        "main": {prov: {n: {k: [float(x) for x in v] for k, v in r.items()}
                        for n, r in res.items()}
                 for prov, res in results_by_provider.items()},
        "llm_stats": llm_stats_by_provider,
        "ablation_persona": persona_abl,
        "ablation_nsim": {str(k): v for k, v in nsim_abl.items()},
        "cluster_info": cluster_info,
    }
    out_path = out_dir / "experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
