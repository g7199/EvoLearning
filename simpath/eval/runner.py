"""Experiment runner: train, evaluate, save results."""
import os, json, time, subprocess, sys
import numpy as np
import torch
import pickle
from simpath.eval.config import ExperimentConfig, get_dataset_config
from simpath.eval.data import load_data
from simpath.eval.kes import KES, load_dkt
from simpath.eval.graph import load_graph
from simpath.eval.methods import METHOD_REGISTRY, list_methods

# Import all methods to trigger registration
import simpath.eval.methods.random_method
import simpath.eval.methods.target_repeat
import simpath.eval.methods.rule_based
import simpath.eval.methods.gru4rec
import simpath.eval.methods.ppo_vanilla
import simpath.eval.methods.evolearning
import simpath.eval.methods.cseal
import simpath.eval.methods.dlelp
import simpath.eval.methods.knowlp
import simpath.eval.methods.gehrl


def run_single_method(config: ExperimentConfig, method_name: str, device: str):
    """Train and evaluate a single method. Returns test results dict."""
    ds = get_dataset_config(config.dataset)
    print(f"\n{'='*60}")
    print(f"  {method_name} | {config.dataset} | L={config.L} | seed={config.seed} | {device}")
    print(f"{'='*60}")

    # Load data (uses FIXED data seed internally — same split for all experiments)
    train_data, val_data, test_data = load_data(ds)

    # Set training seeds AFTER data loading (controls model init + training randomness)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    print(f"  Data: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    # Load DKT
    dkt, skill_map = load_dkt(ds, device)
    kes = KES(dkt, ds['num_c'], device, ds['max_hist'])

    # Load graph
    method_cls = METHOD_REGISTRY[method_name]
    graph = None
    if method_cls.needs_graph:
        graph_type = getattr(method_cls, 'graph_type', 'dkt')
        graph = load_graph(ds, graph_type)
        print(f"  Graph ({graph_type}): {int(graph.prereq.sum())} prereq, {int(graph.sim.sum())//2} sim")

    # Load experts
    experts = None
    if method_cls.needs_experts:
        evo_path = ds['evo_path_template'].format(L=config.L)
        if os.path.exists(evo_path):
            with open(evo_path, 'rb') as f:
                experts = pickle.load(f)
            print(f"  Experts: {len(experts)} from {evo_path} "
                  f"(path_len={len(experts[0][2]) if experts else 0})")
        else:
            print(f"  WARNING: Expert file {evo_path} not found!")
            print(f"  Run: python scripts/setup_pipeline.py --dataset {config.dataset} --L {config.L} --gpu 0")

    # Output directory
    out_dir = os.path.join(config.save_dir, f"{config.dataset}_L{config.L}_seed{config.seed}",
                           method_name)
    os.makedirs(out_dir, exist_ok=True)

    # Create method
    method = method_cls(ds['num_c'], config.L, ds['hidden'], device)

    # Train (always call — even no-training methods use it to set graph etc.)
    t0 = time.time()
    method.train(train_data, val_data, kes, graph, experts,
                 n_episodes=config.n_episodes,
                 val_interval=config.val_interval,
                 out_dir=out_dir)

    # Save latest checkpoint
    method.save(os.path.join(out_dir, 'latest_model.pt'))

    # If best_state exists, save best checkpoint and evaluate both
    elapsed = time.time() - t0

    # ═══ Final Report: best + latest, val + test ═══
    print(f"\n  Generating final report...", flush=True)
    report = {
        'method': method_name,
        'dataset': config.dataset,
        'L': config.L,
        'seed': config.seed,
        'train_time_sec': elapsed,
        'n_train': len(train_data),
        'n_val': len(val_data),
        'n_test': len(test_data),
    }

    # Latest model evaluation
    latest_val = kes.evaluate_batch(val_data, lambda m, t, k, hc, hr: method.predict(m, t, k, hc, hr))
    latest_test = kes.evaluate_batch(test_data, lambda m, t, k, hc, hr: method.predict(m, t, k, hc, hr))
    report['latest_val_ep'] = float(np.mean(latest_val))
    report['latest_test_ep'] = float(np.mean(latest_test))
    report['latest_test_std'] = float(np.std(latest_test))
    report['latest_test_median'] = float(np.median(latest_test))
    report['latest_test_per_student'] = [float(e) for e in latest_test]

    # Best model evaluation (if available)
    best_path = os.path.join(out_dir, 'best_model.pt')
    if hasattr(method, '_best_state') and method._best_state:
        # Save best checkpoint
        if hasattr(method, 'policy'):
            method.policy.load_state_dict(method._best_state)
        elif hasattr(method, 'high'):  # GEHRL
            method.high.load_state_dict(method._best_h)
            method.low.load_state_dict(method._best_l)
        method.save(best_path)

        best_val = kes.evaluate_batch(val_data, lambda m, t, k, hc, hr: method.predict(m, t, k, hc, hr))
        best_test = kes.evaluate_batch(test_data, lambda m, t, k, hc, hr: method.predict(m, t, k, hc, hr))
        report['best_val_ep'] = float(np.mean(best_val))
        report['best_test_ep'] = float(np.mean(best_test))
        report['best_test_std'] = float(np.std(best_test))
        report['best_test_median'] = float(np.median(best_test))
        report['best_test_per_student'] = [float(e) for e in best_test]
    else:
        # No training → latest = best
        report['best_val_ep'] = report['latest_val_ep']
        report['best_test_ep'] = report['latest_test_ep']
        report['best_test_std'] = report['latest_test_std']
        report['best_test_median'] = report['latest_test_median']
        report['best_test_per_student'] = report['latest_test_per_student']
        method.save(best_path)

    # Save report
    with open(os.path.join(out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  {'─'*50}")
    print(f"  {method_name} FINAL REPORT")
    print(f"  {'─'*50}")
    print(f"  Best Val EP:     {report['best_val_ep']:+.4f}")
    print(f"  Best Test EP:    {report['best_test_ep']:+.4f} +/- {report.get('best_test_std', 0):.4f}")
    print(f"  Latest Val EP:   {report['latest_val_ep']:+.4f}")
    print(f"  Latest Test EP:  {report['latest_test_ep']:+.4f} +/- {report.get('latest_test_std', 0):.4f}")
    print(f"  Time: {elapsed:.0f}s")
    return report


def run_all_methods(config: ExperimentConfig):
    """Run all methods in parallel with tqdm monitoring."""
    from tqdm import tqdm
    gpus = [g.strip() for g in config.gpu.split(',')]
    methods = list_methods()
    base_dir = os.path.join(config.save_dir, f"{config.dataset}_L{config.L}_seed{config.seed}")

    # No-training methods first (instant)
    no_train = [m for m in methods if not METHOD_REGISTRY[m].needs_training]
    train_methods = [m for m in methods if METHOD_REGISTRY[m].needs_training]

    all_results = {}
    print(f"\n  Running {len(no_train)} no-training methods...")
    for m in no_train:
        r = run_single_method(config, m, f'cuda:{gpus[0]}')
        all_results[m] = r

    # Launch all training methods as subprocesses (round-robin GPU assignment)
    print(f"\n  Launching {len(train_methods)} training methods in parallel...")
    procs = {}
    for i, m in enumerate(train_methods):
        gpu = gpus[i % len(gpus)]
        out_dir = os.path.join(base_dir, m)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [
            sys.executable, '-m', 'simpath.eval.runner',
            '--dataset', config.dataset, '--method', m,
            '--L', str(config.L), '--seed', str(config.seed),
            '--gpu', gpu, '--save_dir', config.save_dir,
            '--n_episodes', str(config.n_episodes),
        ]
        log_path = os.path.join(out_dir, 'train.log')
        # Clear old progress so tqdm doesn't show stale data
        prog_path = os.path.join(out_dir, 'progress.json')
        if os.path.exists(prog_path):
            os.remove(prog_path)
        log_f = open(log_path, 'w')
        p = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        procs[m] = {'proc': p, 'gpu': gpu, 'log': log_f, 'out_dir': out_dir}
        print(f"    {m:<20} → GPU:{gpu} (PID {p.pid})")

    # tqdm monitoring loop
    print()
    bars = {}
    for m in train_methods:
        bars[m] = tqdm(total=config.n_episodes, desc=f"{m:<16}", position=train_methods.index(m),
                       bar_format='{l_bar}{bar:20}{r_bar}', leave=True)

    import time as _time
    while True:
        all_done = True
        for m in train_methods:
            p = procs[m]['proc']
            progress_path = os.path.join(procs[m]['out_dir'], 'progress.json')
            if p.poll() is None:
                all_done = False
            try:
                with open(progress_path) as f:
                    prog = json.load(f)
                # Dynamic total (GRU4Rec uses 5000 epochs, not n_episodes)
                if prog.get('total') and prog['total'] != bars[m].total:
                    bars[m].total = prog['total']
                bars[m].n = prog.get('ep', 0)
                postfix = {}
                if prog.get('reward') is not None:
                    postfix['R'] = f"{prog['reward']:+.3f}"
                if prog.get('val') is not None:
                    postfix['V'] = f"{prog['val']:+.4f}"
                postfix['gpu'] = procs[m]['gpu']
                bars[m].set_postfix(postfix)
                bars[m].refresh()
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

        if all_done:
            # Final update
            for m in train_methods:
                bars[m].n = bars[m].total
                bars[m].refresh()
            break
        _time.sleep(2)

    for m in train_methods:
        bars[m].close()
    print()

    # Close log files
    for m in train_methods:
        procs[m]['log'].close()

    # Collect results
    for m in train_methods:
        report_path = os.path.join(procs[m]['out_dir'], 'report.json')
        if os.path.exists(report_path):
            with open(report_path) as f:
                all_results[m] = json.load(f)
        else:
            print(f"  WARNING: Report not found for {m} (check {procs[m]['out_dir']}/train.log)")

    # Summary
    summary_path = os.path.join(base_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS: {config.dataset} | L={config.L} | seed={config.seed}")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Method':<20} {'BestTest':>10} {'Std':>8} {'BestVal':>10} {'Time':>8}")
    print(f"{'─'*5} {'─'*20} {'─'*10} {'─'*8} {'─'*10} {'─'*8}")
    sorted_r = sorted(all_results.items(),
                       key=lambda x: x[1].get('best_test_ep', -999), reverse=True)
    for rank, (name, r) in enumerate(sorted_r, 1):
        print(f"{rank:<5} {name:<20} "
              f"{r.get('best_test_ep', 0):>+10.4f} "
              f"{r.get('best_test_std', 0):>8.4f} "
              f"{r.get('best_val_ep', 0):>+10.4f} "
              f"{r.get('train_time_sec', 0):>7.0f}s")
    print(f"{'='*70}")
    print(f"  Saved: {summary_path}")

    return all_results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='assist09')
    p.add_argument('--method', default='all')
    p.add_argument('--L', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', default='0')
    p.add_argument('--save_dir', default='outputs/experiments')
    p.add_argument('--n_episodes', type=int, default=30000)
    args = p.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset, method=args.method, L=args.L,
        seed=args.seed, gpu=args.gpu, save_dir=args.save_dir,
        n_episodes=args.n_episodes)

    if args.method == 'all':
        run_all_methods(config)
    else:
        run_single_method(config, args.method, f'cuda:{args.gpu}')
