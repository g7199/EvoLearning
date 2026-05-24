#!/usr/bin/env python3
"""Ablation: Evolutionary expert quality matters.
Train EVOL-BC on RANDOM-path experts (same pipeline, but path = random instead of evolved).
If random-expert version performs poorly, the evolutionary search is the key driver.
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from simpath.eval.config import ExperimentConfig
from simpath.eval.runner import run_single_method
import simpath.eval.methods.evolearning_bc  # register

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', default='0')
    p.add_argument('--n_episodes', type=int, default=10000)
    args = p.parse_args()

    # Patch evo_path_template to random_experts
    from simpath.eval.config import DATASET_CONFIGS
    DATASET_CONFIGS[args.dataset]['evo_path_template'] = \
        f'outputs/random_experts_{args.dataset}_L{{L}}.pkl'

    config = ExperimentConfig(
        dataset=args.dataset, method='EvoLearning-BC',
        L=args.L, seed=args.seed, gpu=args.gpu,
        save_dir='outputs/ablation_random_expert',
        n_episodes=args.n_episodes
    )
    run_single_method(config, 'EvoLearning-BC', f'cuda:{args.gpu}')
