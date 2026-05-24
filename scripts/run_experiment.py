#!/usr/bin/env python3
"""
EvoLearning — Unified Experiment Runner.

Usage:
  # Single method
  python scripts/run_experiment.py --dataset assist09 --method EvoLearning --L 5 --seed 42 --gpu 0

  # All methods (sequential on 1 GPU)
  python scripts/run_experiment.py --dataset assist09 --method all --L 5 --seed 42 --gpu 0

  # All methods (parallel on 2 GPUs)
  python scripts/run_experiment.py --dataset assist09 --method all --L 5 --seed 42 --gpu 0,1

  # 3-seed experiment
  for seed in 42 123 7; do
    python scripts/run_experiment.py --dataset junyi --method all --L 5 --seed $seed --gpu 0,1
  done

Methods: EvoLearning, PPO-vanilla, CSEAL, DLELP, GEHRL, KnowLP, GRU4Rec, Rule-based, Random, Target-repeat
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import warnings
warnings.filterwarnings('ignore')

from simpath.eval.config import ExperimentConfig
from simpath.eval.runner import run_single_method, run_all_methods
from simpath.eval.methods import list_methods


def main():
    p = argparse.ArgumentParser(description="EvoLearning Experiment Runner")
    p.add_argument('--dataset', default='assist09', choices=['assist09', 'assist15', 'junyi', 'ednet'])
    p.add_argument('--method', default='all',
                   help='Method name or "all". Available: ' + ', '.join(list_methods()))
    p.add_argument('--L', type=int, default=5, help='Path length (5, 10, 20)')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--gpu', default='0', help='GPU device(s): "0", "1", "0,1"')
    p.add_argument('--save_dir', default='outputs/experiments')
    p.add_argument('--n_episodes', type=int, default=30000, help='RL training episodes')
    p.add_argument('--val_interval', type=int, default=2000)
    args = p.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset, method=args.method, L=args.L,
        seed=args.seed, gpu=args.gpu, save_dir=args.save_dir,
        n_episodes=args.n_episodes, val_interval=args.val_interval)

    print(f"{'='*60}")
    print(f"  EvoLearning Experiment")
    print(f"  Dataset: {config.dataset} | L={config.L} | Seed={config.seed}")
    print(f"  GPU: {config.gpu} | Episodes: {config.n_episodes}")
    print(f"  Methods: {config.method}")
    print(f"{'='*60}")

    if args.method == 'all':
        run_all_methods(config)
    else:
        run_single_method(config, args.method, f'cuda:{args.gpu}')


if __name__ == '__main__':
    main()
