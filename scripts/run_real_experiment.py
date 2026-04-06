#!/usr/bin/env python3
"""
SimPath experiment on real datasets (EdNet KT3 / ASSISTments 2015).

Usage:
    # Step 1: Download data
    python scripts/run_real_experiment.py download --dataset assistments

    # Step 2: Preprocess
    python scripts/run_real_experiment.py preprocess --dataset assistments

    # Step 3: Train KT models
    python scripts/run_real_experiment.py train --dataset assistments

    # Step 4: Run experiment
    python scripts/run_real_experiment.py run --dataset assistments --provider openai

    # All at once:
    python scripts/run_real_experiment.py all --dataset assistments --provider openai
"""

import sys, os, argparse, json, pickle
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["download", "preprocess", "train", "run", "all"])
    p.add_argument("--dataset", choices=["ednet", "assistments"], default="assistments")
    p.add_argument("--provider", default="mock", choices=["mock", "openai", "anthropic", "both"])
    p.add_argument("--n_test", type=int, default=200, help="Number of test students")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-ablation", dest="no_ablation", action="store_true")
    p.add_argument("--quick", action="store_true", help="Small subset for sanity check")
    p.add_argument("--output_dir", default="outputs/results")
    return p.parse_args()


def stage_download(args):
    from simpath.data.download import download_ednet, download_assistments
    if args.dataset == "ednet":
        download_ednet()
    else:
        download_assistments()


def stage_preprocess(args):
    from simpath.data.preprocess import preprocess_ednet, preprocess_assistments
    if args.dataset == "ednet":
        preprocess_ednet("data/raw/ednet")
    else:
        preprocess_assistments("data/raw/assistments/2015_100_skill_builders_main_problems.csv")


def stage_train(args):
    from simpath.data.dataset import KTDataset
    from simpath.kt.dkt import DKT
    from simpath.kt.akt import AKT
    from simpath.kt.saint import SAINT
    from simpath.kt.train import train_kt_model
    from simpath.utils.config import load_config

    cfg = load_config()
    data = _load_processed(args.dataset)
    students = data["students"]
    n_q = data["n_questions"]
    n_kc = data["n_kcs"]

    print(f"\n[Train] Dataset={args.dataset}, Q={n_q}, KC={n_kc}, Students={len(students)}")

    # Build datasets
    train_ds = KTDataset(students, "train", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    val_ds = KTDataset(students, "val", max_seq_len=200, n_questions=n_q, n_kcs=n_kc)
    print(f"  Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    # Train AKT (recommendation model)
    print(f"\n{'='*50}")
    print("Training AKT (recommendation model)...")
    print(f"{'='*50}")
    akt = AKT(n_q, n_kc, **cfg["kt"]["akt"])
    akt, akt_hist = train_kt_model(
        akt, train_ds, val_ds, model_name="akt",
        dataset_name=args.dataset, device=args.device,
        **cfg["kt"]["training"],
    )

    # Train SAINT (evaluation model) — separate from AKT [Defense: #1]
    print(f"\n{'='*50}")
    print("Training SAINT (evaluation model)...")
    print(f"{'='*50}")
    saint = SAINT(n_q, n_kc, **cfg["kt"]["saint"])
    saint, saint_hist = train_kt_model(
        saint, train_ds, val_ds, model_name="saint",
        dataset_name=args.dataset, device=args.device,
        **cfg["kt"]["training"],
    )

    # Train DKT (for baselines)
    print(f"\n{'='*50}")
    print("Training DKT (for baselines)...")
    print(f"{'='*50}")
    dkt = DKT(n_q, n_kc, **cfg["kt"]["dkt"])
    dkt, dkt_hist = train_kt_model(
        dkt, train_ds, val_ds, model_name="dkt",
        dataset_name=args.dataset, device=args.device,
        **cfg["kt"]["training"],
    )

    print("\n[Train] All models trained.")


def stage_run(args):
    from simpath.utils.seeds import set_global_seed
    from simpath.utils.config import load_config
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

    cfg = load_config()
    set_global_seed(args.seed)
    data = _load_processed(args.dataset)

    students = data["students"]
    questions = data["questions"]
    s2q = data["skill_to_questions"]

    # Select test students (last n_test with enough held_out data)
    eligible = [s for s in students if len(s["held_out"]) >= 5]
    if args.quick:
        test_students = eligible[-10:]
        cfg["personas"]["N_sim"] = 3
        cfg["paths"]["K"] = 3
    else:
        test_students = eligible[-min(args.n_test, len(eligible)):]

    print(f"\n[Run] Dataset={args.dataset}, Test students={len(test_students)}, "
          f"Provider={args.provider}")

    L = cfg["paths"]["L"]
    K = cfg["paths"]["K"]
    N_sim = cfg["personas"]["N_sim"]
    weights = (cfg["selection"]["weights"]["w1"],
               cfg["selection"]["weights"]["w2"],
               cfg["selection"]["weights"]["w3"])

    # Build path generator
    if args.provider == "mock":
        def gen(mastery, pool, K=K, L=L, **kw):
            return mock_generate_paths(mastery, pool, K, L), {"pool_hit_rate": 1.0, "fill_rate": 0.0, "constraint_violation_rate": 0.0}
    else:
        from simpath.llm.client import LLMClient
        from simpath.paths.ordering import generate_candidate_paths_llm

        providers_to_run = []
        if args.provider == "both":
            providers_to_run = [
                ("openai", None, "gpt-5.4-2026-03-05"),
                ("anthropic", None, "claude-sonnet-4-6"),
            ]
        else:
            providers_to_run = [(args.provider, None, None)]

    # Run per provider
    all_provider_results = {}

    provider_list = [("mock", None, None)] if args.provider == "mock" else providers_to_run

    for prov_name, prov_key, prov_model in provider_list:
        print(f"\n{'='*60}")
        print(f"Running with provider={prov_name}")
        print(f"{'='*60}")

        if prov_name == "mock":
            path_gen = lambda mastery, pool, K=K, L=L, **kw: (
                mock_generate_paths(mastery, pool, K, L),
                {"pool_hit_rate": 1.0, "fill_rate": 0.0, "constraint_violation_rate": 0.0},
            )
        else:
            client = LLMClient(provider=prov_name, model=prov_model, api_key=prov_key)
            print(f"  LLM: {client.provider}/{client.model}")
            def path_gen(mastery, pool, K=K, L=L, client=client, **kw):
                return generate_candidate_paths_llm(mastery, pool, client, K, L, **kw)

        methods_results = {
            "SimPath (minimax)": defaultdict(list),
            "B1: DKT+Random": defaultdict(list),
            "B2: DKT+RuleBased": defaultdict(list),
            "B6: SimPath-NoRobust": defaultdict(list),
        }
        ablation_strategy = {n: defaultdict(list) for n in
            ["minimax_regret", "average", "maximin", "hurwicz_0.3", "hurwicz_0.7", "weighted_minimax"]}
        llm_stats_list = []

        for idx, student in enumerate(test_students):
            if idx % 20 == 0:
                print(f"  [{prov_name}] Student {idx+1}/{len(test_students)}...")

            mastery = student["mastery"]
            sid = student["student_id"]
            theta_real = extract_realistic_params(student["features"])
            target_kcs = [kc for kc, m in mastery.items() if m < 0.6] or list(mastery.keys())[:5]
            eval_kw = dict(theta_real=theta_real, target_kcs=target_kcs,
                           weights=weights, N_sim=N_sim)

            pool = retrieve_candidate_pool(mastery, questions, s2q, L=L)
            candidate_paths, llm_stats = path_gen(mastery, pool, K=K, L=L, seed_base=42+idx)
            llm_stats_list.append(llm_stats)

            S = simulate_all_personas(mastery, candidate_paths, theta_real, sid, N_sim, target_kcs)
            phi = composite_scores(S, weights)
            best_idx, _, _ = minimax_regret_select(phi)

            # SimPath
            m = evaluate_path(mastery=mastery, path=candidate_paths[best_idx],
                              student_id=sid, **eval_kw)
            for k, v in m.items():
                methods_results["SimPath (minimax)"][k].append(v)

            # Selection strategy ablation
            for sname, sidx in [
                ("minimax_regret", best_idx), ("average", average_select(phi)),
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
                ("B1: DKT+Random", recommend_random(mastery, s2q, L=L)),
                ("B2: DKT+RuleBased", recommend_rulebased(mastery, s2q, L=L)),
            ]:
                m = evaluate_path(mastery=mastery, path=bpath,
                                  student_id=f"{sid}_{bname}", **eval_kw)
                for k, v in m.items():
                    methods_results[bname][k].append(v)

            # B6
            norobust_idx = average_select(phi)
            m = evaluate_path(mastery=mastery, path=candidate_paths[norobust_idx],
                              student_id=f"{sid}_b6", **eval_kw)
            for k, v in m.items():
                methods_results["B6: SimPath-NoRobust"][k].append(v)

        # Aggregate LLM stats
        agg_stats = {}
        for key in ["pool_hit_rate", "fill_rate", "constraint_violation_rate"]:
            vals = [s.get(key, 0) for s in llm_stats_list if s]
            agg_stats[key] = float(np.mean(vals)) if vals else 0

        all_provider_results[prov_name] = {
            "methods": methods_results,
            "ablation_strategy": ablation_strategy,
            "llm_stats": agg_stats,
        }

    # Print results
    _print_results(all_provider_results, args.dataset)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.provider}_results.json"
    _save_results(all_provider_results, out_path, args.dataset)
    print(f"\nResults saved to {out_path}")


def _load_processed(dataset):
    path = Path(f"data/processed/{dataset}")
    pkl = list(path.glob("*.pkl"))
    if not pkl:
        print(f"Error: No processed data found in {path}/")
        print(f"Run: python scripts/run_real_experiment.py preprocess --dataset {dataset}")
        sys.exit(1)
    with open(pkl[0], "rb") as f:
        return pickle.load(f)


def _print_results(all_results, dataset_name):
    from simpath.evaluation.statistical_tests import paired_comparison

    for prov, data in all_results.items():
        results = data["methods"]
        stats = data["llm_stats"]

        print(f"\n{'='*70}")
        print(f"RESULTS — {dataset_name.upper()} / {prov.upper()}")
        if stats:
            print(f"  LLM: pool_hit={stats.get('pool_hit_rate',0):.3f}, "
                  f"fill={stats.get('fill_rate',0):.3f}, "
                  f"violation={stats.get('constraint_violation_rate',0):.3f}")
        print(f"{'='*70}")

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

        # Selection ablation
        abl = data["ablation_strategy"]
        print(f"\n--- Selection Strategy Ablation ---")
        print(f"{'Strategy':<20} {'MG':>8} {'SR':>8} {'CE':>8} {'RI':>8}")
        print("-" * 55)
        for sn, sr in abl.items():
            print(f"{sn:<20} {np.mean(sr['MG']):8.4f} {np.mean(sr['SR']):8.4f} "
                  f"{np.mean(sr['CE']):8.4f} {np.mean(sr['RI']):8.4f}")


def _save_results(all_results, path, dataset_name):
    save = {"dataset": dataset_name}
    for prov, data in all_results.items():
        save[prov] = {
            "methods": {n: {k: [float(x) for x in v] for k, v in r.items()}
                        for n, r in data["methods"].items()},
            "ablation_strategy": {n: {k: [float(x) for x in v] for k, v in r.items()}
                                  for n, r in data["ablation_strategy"].items()},
            "llm_stats": data["llm_stats"],
        }
    with open(path, "w") as f:
        json.dump(save, f, indent=2)


def main():
    args = parse_args()
    if args.stage == "all":
        stage_download(args)
        stage_preprocess(args)
        stage_train(args)
        stage_run(args)
    elif args.stage == "download":
        stage_download(args)
    elif args.stage == "preprocess":
        stage_preprocess(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "run":
        stage_run(args)


if __name__ == "__main__":
    main()
