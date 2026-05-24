# EvoLearning

**Evolutionary Expert-Guided Reinforcement Learning for Learning Path Recommendation**

## Overview

EvoLearning recommends personalized learning paths by combining evolutionary expert generation with behavioral cloning and PPO fine-tuning. This repository provides a unified evaluation framework comparing 11 methods (Table 1) across 3 datasets.

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

The three datasets from the paper. "Learners" is the number used in the
learning-path experiments (train + val + test split).

| Dataset | Learners | KCs | Source |
| --------- | -------- | ----- | ------ |
| Junyi | 5,000 | 39 | Junyi Academy (PSLC DataShop 1198) |
| ASSIST15 | 7,284 | 100 | ASSISTments 2015 (100 skill builders) |
| EdNet | 24,850 | 189 | EdNet KT1 (Riiid) |

## Setup

### 1. Environment

```bash
# Python 3.10+ (reference environment: 3.11)
python3 -m venv .venv

# Install deps into the venv WITHOUT activating it (activation is a
# shell-state hack that differs across shells/OSes); call the venv
# interpreter directly instead:
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# Then run scripts with that same interpreter, e.g.:
#   .venv/bin/python scripts/run_experiment.py --dataset junyi --L 5
#
# (If you prefer activation: 'source .venv/bin/activate' once, then
#  plain 'python ...'. On Windows: .venv\Scripts\python.exe ...)
```

Dependencies are pinned in `requirements.txt` (mirrors `pyproject.toml`).
The scripts add the repo root to `sys.path`, so no editable install is
needed. To reproduce **only** the main results table (Table 1) with no
GPU, no training and no third-party packages, see
[`repro_main_table/`](repro_main_table/) -- it is pure-stdlib and runs
straight out of a clean checkout.

### 2. Get the data (download it yourself, then place as shown)

Raw datasets are NOT bundled (licensing and size). Download each one from
its source and place the files at the exact paths below; preprocessing
reads these locations.

```
# Junyi Academy   (PSLC DataShop dataset 1198)
#   https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198
data/raw/junyi/junyi_extracted/junyi_ProblemLog_original.csv
data/raw/junyi/junyi_extracted/junyi_Exercise_table.csv

# ASSIST15  (ASSISTments 2015, "100 skill builders")
#   https://sites.google.com/site/assistmentsdata/
data/raw/assistments/2015_100_skill_builders_main_problems.csv

# EdNet KT1  (Riiid)
#   https://github.com/riiid/ednet
data/raw/ednet/KT1/u1.csv , u2.csv , ...   (per-student interaction files)
data/raw/ednet/contents/questions.csv
```

For junyi, a helper can fetch it automatically:

```bash
.venv/bin/python scripts/download_data.py --dataset junyi
```

### 3. Pipeline (preprocessing → DKT → graph → experts)

```bash
.venv/bin/python scripts/setup_pipeline.py --dataset junyi    --gpu 0
.venv/bin/python scripts/setup_pipeline.py --dataset assist15 --gpu 0
.venv/bin/python scripts/setup_pipeline.py --dataset ednet    --gpu 0
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
python scripts/run_experiment.py --dataset junyi --method EvoLearning --L 5 --seed 42 --gpu 0
```

### All methods (parallel on 2 GPUs)
```bash
python scripts/run_experiment.py --dataset junyi --method all --L 5 --seed 42 --gpu 0,1
```

### 3-seed experiment
```bash
for seed in 42 123 7; do
  python scripts/run_experiment.py --dataset junyi --method all --L 5 --seed $seed --gpu 0,1
done
```

### Main results table (Table 1)

After the runs above finish, aggregate the per-run reports into the table:

```bash
.venv/bin/python scripts/aggregate_main_table.py
```

This reads outputs/experiments/<dataset>_L<L>_seed<seed>/<method>/report.json
across the three seeds and writes outputs/main_table.txt, .csv and .tex.
For an instant, dependency-free reproduction of Table 1 from pre-computed
numbers (no data, GPU or training), run repro_main_table/ instead.

### Parameters
| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | junyi | Dataset: `junyi`, `assist15`, `ednet` |
| `--method` | all | Method name or `all` |
| `--L` | 5 | Path length: 5, 10, 20 |
| `--seed` | 42 | Training seed (data split is fixed) |
| `--gpu` | 0 | GPU(s): `0`, `1`, `0,1` |
| `--n_episodes` | 30000 | RL training episodes |
| `--val_interval` | 2000 | Validate every N episodes |

## Output Structure

```
outputs/experiments/
  assist15_L10_seed42/
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
