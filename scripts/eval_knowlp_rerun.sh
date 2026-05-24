#!/bin/bash
# Re-evaluate KnowLP with correct graph_type='knowlp' (now patched in eval scripts).
PY=/home/guest/miniconda3/envs/sim/bin/python
EVAL_CROSS=/home/guest/sim/scripts/eval_cross_sim_all.py
EVAL_STOCH=/home/guest/sim/scripts/eval_stochastic_all_methods.py
CKPT_DIR=/home/guest/sim/outputs/checkpoints

# Cross-DKT: 4 alts × 3 seeds = 12 runs (seed 42 alt_s7 already done)
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s100_assist15.pt --alt_name alt_s100 --seed 42 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s200_assist15.pt --alt_name alt_s200 --seed 42 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s333_assist15.pt --alt_name alt_s333 --seed 42 --gpu 0

$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s7_assist15.pt   --alt_name alt_s7   --seed 123 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s100_assist15.pt --alt_name alt_s100 --seed 123 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s200_assist15.pt --alt_name alt_s200 --seed 123 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s333_assist15.pt --alt_name alt_s333 --seed 123 --gpu 0

$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s7_assist15.pt   --alt_name alt_s7   --seed 7 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s100_assist15.pt --alt_name alt_s100 --seed 7 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s200_assist15.pt --alt_name alt_s200 --seed 7 --gpu 0
$PY $EVAL_CROSS --method KnowLP --alt_ckpt $CKPT_DIR/pykt_dkt_alt_s333_assist15.pt --alt_name alt_s333 --seed 7 --gpu 0

# Stochastic: 3 seeds × 4 response models (script handles all 4 internally) = 3 runs
$PY $EVAL_STOCH --method KnowLP --seed 42  --gpu 0 --n_students 200 --n_rollouts 20
$PY $EVAL_STOCH --method KnowLP --seed 123 --gpu 0 --n_students 200 --n_rollouts 20
$PY $EVAL_STOCH --method KnowLP --seed 7   --gpu 0 --n_students 200 --n_rollouts 20

echo "=== KnowLP RE-EVAL DONE ==="
