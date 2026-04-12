# EvoLearning

**Evolutionary Expert-Guided Reinforcement Learning for Learning Path Recommendation**

## Overview

EvoLearning recommends personalized learning paths by combining evolutionary expert generation with behavioral cloning and PPO fine-tuning. This repository provides a unified evaluation framework comparing 10 methods across 2 datasets.

## Methods (10)

| Method | Type | Source |
|--------|------|--------|
| **EvoLearning** | Evo → BC → PPO | Ours |
| PPO-vanilla | RL (PPO) | Baseline |
| CSEAL | A2C + graph masking | Cognitive Navigation |
| DLELP | PPO + S-Agent + A* | arXiv 2506.22303 |
| GEHRL | Hierarchical RL | Chen et al., CIKM 2023 |
| KnowLP | DLELP + EDU-GraphRAG | Cheng et al., 2025 |
| GRU4Rec | GRU autoregressive BC | Hidasi et al., 2015 |
| Rule-based | Graph topological sort | Baseline |
| Random | Random concept selection | Baseline |
| Target-repeat | Repeat target concepts | Baseline |

## Datasets

| Dataset | Students | KCs | Source |
|---------|----------|-----|--------|
| ASSIST09 | 2,303 | 123 | [ASSISTments](https://sites.google.com/site/assistmentsdata/) |
| Junyi | 10,000 | 39 | [Junyi Academy](https://pslcdatashop.web.cmu.edu/) |

## Setup

### 1. Environment

```bash
conda create -n sim python=3.11
conda activate sim
pip install -e .
```

### 2. Data

Download raw datasets and place them:
```
data/raw/assist09/skill_builder_data_corrected.csv
data/raw/junyi/junyi_extracted/junyi_ProblemLog_original.csv
data/raw/junyi/junyi_extracted/junyi_Exercise_table.csv
```

### 3. Pipeline (preprocessing → DKT → graph → experts)

```bash
python scripts/setup_pipeline.py --dataset assist09 --gpu 0
python scripts/setup_pipeline.py --dataset junyi --gpu 1
```

This generates:
- Preprocessed data (`data/processed/`)
- DKT checkpoints (`outputs/checkpoints/`)
- DKT influence graphs (`outputs/concept_graph_dkt_*.pkl`)
- Evo expert trajectories (`outputs/evo_dpk5_*.pkl`)
- KnowLP EDU-GraphRAG graph (requires `OPENAI_API_KEY` in `.env`)

### 4. KnowLP (optional)

KnowLP requires an OpenAI API key for EDU-GraphRAG graph generation:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

## Running Experiments

### Single method
```bash
python scripts/run_experiment.py --dataset assist09 --method EvoLearning --L 5 --seed 42 --gpu 0
```

### All methods (parallel on 2 GPUs)
```bash
python scripts/run_experiment.py --dataset assist09 --method all --L 5 --seed 42 --gpu 0,1
```

### 3-seed experiment
```bash
for seed in 42 123 7; do
  python scripts/run_experiment.py --dataset assist09 --method all --L 5 --seed $seed --gpu 0,1
done
```

### Parameters
| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | assist09 | Dataset: `assist09`, `junyi` |
| `--method` | all | Method name or `all` |
| `--L` | 5 | Path length: 5, 10, 20 |
| `--seed` | 42 | Training seed (data split is fixed) |
| `--gpu` | 0 | GPU(s): `0`, `1`, `0,1` |
| `--n_episodes` | 30000 | RL training episodes |
| `--val_interval` | 2000 | Validate every N episodes |

## Output Structure

```
outputs/experiments/
  assist09_L5_seed42/
    EvoLearning/
      progress.json        # Real-time progress (for tqdm)
      train.log            # Training log
      best_model.pt        # Best validation checkpoint
      latest_model.pt      # Final checkpoint
      checkpoint_ep2000.pt # Periodic checkpoints
      report.json          # Final report (best/latest val+test EP)
    PPO-vanilla/
      ...
    results_summary.json   # All methods compared
```

## Evaluation

- **KES (Knowledge Evolution Simulator)**: DKT-based environment with deterministic threshold (P > 0.5 → correct)
- **EP (Effectiveness Percentage)**: `mean((E_end - E_start) / (1 - E_start))` over target concepts
- **Data split**: 50% dataSim (DKT) / 50% dataOff (RL), fixed seed=42

## Citation

```bibtex
@article{evolearning2025,
  title={EvoLearning: Evolutionary Expert-Guided Reinforcement Learning for Learning Path Recommendation},
  year={2025}
}
```
