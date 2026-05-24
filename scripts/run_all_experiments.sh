#!/bin/bash
# Junyi 전체 실험: 3 seeds × 3 L values = 9 runs
# GPU 0,1 병렬 사용

set -e
DATASET="junyi"
GPUS="0,1"
EPISODES=10000

for L in 5 10 20; do
    # Evo experts 생성
    python scripts/setup_pipeline.py --dataset ${DATASET} --gpu 0 --L ${L} --force

    # 3 seeds 실험
    for SEED in 42 123 7; do
        echo ""
        echo "========================================"
        echo "  ${DATASET} | L=${L} | seed=${SEED}"
        echo "========================================"
        python scripts/run_experiment.py \
            --dataset ${DATASET} \
            --method all \
            --L ${L} \
            --seed ${SEED} \
            --gpu ${GPUS} \
            --n_episodes ${EPISODES}
    done
done

echo ""
echo "========================================"
echo "  ALL DONE!"
echo "========================================"
