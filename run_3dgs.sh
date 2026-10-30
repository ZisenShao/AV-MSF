#!/bin/bash
set -euo pipefail

# Train 3DGS.
#
# Usage:
#   bash run_3dgs.sh
#   CUDA_VISIBLE_DEVICES=1 bash run_3dgs.sh    # pick a GPU

OBJECT_IDS=(
    1 2 25 41 44 52 57 75 80 100
)

cd "$(dirname "$0")"

for OBJECT_ID in "${OBJECT_IDS[@]}"; do
    echo "Training 3DGS for object ${OBJECT_ID}"
    python vision/examples/trainer.py default \
        --data-dir "vision/data/outputs/${OBJECT_ID}" \
        --result-dir "results/3dgs/${OBJECT_ID}" \
        --max-steps 30000 \
        --save-steps 30000 \
        --eval-steps 30000 \
        --no-random-bkgd \
        --no-normalize-world-space
done
