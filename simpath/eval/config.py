"""Experiment and dataset configuration."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    dataset: str = 'assist15'       # 'assist15' | 'junyi' | 'ednet'
    method: str = 'all'             # method name or 'all'
    L: int = 5                      # path length: 5, 10, 20
    seed: int = 42                  # controls data split + model init + training
    gpu: str = '0'                  # GPU device(s): '0', '1', '0,1'
    n_episodes: int = 30000         # RL training episodes
    val_interval: int = 2000        # validate every N episodes
    save_dir: str = 'outputs/experiments'


DATASET_CONFIGS = {
    'assist09': {
        'num_c': 123,
        'hidden': 512,
        'dkt_path': 'outputs/checkpoints/pykt_dkt_best_assist09.pt',
        'dkt_emb': 200,
        'graph_dkt_path': 'outputs/concept_graph_dkt_assist09.pkl',
        'graph_llm_path': 'outputs/concept_graph_assist09.pkl',
        'graph_thr': 0.25,
        'data_path': 'data/processed/assist09/assist09_processed.pkl',
        'evo_path_template': 'outputs/evo_dpk5_assist09_L{L}.pkl',
        'max_hist': 200,
    },
    'assist15': {
        'num_c': 100,
        'hidden': 512,
        'dkt_path': 'outputs/checkpoints/pykt_dkt_best_assist15.pt',
        'dkt_emb': 200,
        'graph_dkt_path': 'outputs/concept_graph_dkt_assist15.pkl',
        'graph_llm_path': 'outputs/knowlp_graph_assist15.pkl',
        'graph_thr': 0.25,
        'data_path': 'data/processed/assistments/assistments_processed.pkl',
        'evo_path_template': 'outputs/evo_dpk5_assist15_L{L}.pkl',
        'max_hist': 200,
    },
    'junyi': {
        'num_c': 39,
        'hidden': 256,
        'dkt_path': 'outputs/checkpoints/pykt_dkt_best_junyi.pt',
        'dkt_emb': 256,
        'graph_dkt_path': 'outputs/concept_graph_dkt_junyi.pkl',
        'graph_llm_path': None,
        'graph_thr': 0.05,
        'data_path': 'data/processed/junyi/junyi_processed.pkl',
        'evo_path_template': 'outputs/evo_dpk5_junyi_L{L}.pkl',
        'max_hist': 200,
    },
    'ednet': {
        'num_c': 189,
        'hidden': 512,
        'dkt_path': 'outputs/checkpoints/pykt_dkt_best_ednet.pt',
        'dkt_emb': 200,
        'graph_dkt_path': 'outputs/concept_graph_dkt_ednet.pkl',
        'graph_llm_path': None,
        'graph_thr': 0.15,
        'data_path': 'data/processed/ednet/ednet_processed.pkl',
        'evo_path_template': 'outputs/evo_dpk5_ednet_L{L}.pkl',
        'max_hist': 200,
    },
}


def get_dataset_config(dataset: str) -> dict:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset]
