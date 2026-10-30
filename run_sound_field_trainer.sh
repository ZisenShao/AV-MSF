#!/bin/bash
set -euo pipefail

# Train modal sound fields
# Usage:
#   bash run_sound_field_trainer.sh                      # configs/train.yaml
#   bash run_sound_field_trainer.sh configs/train.yaml
#   CUDA_VISIBLE_DEVICES=1 bash run_sound_field_trainer.sh

OBJECT_IDS=(
    1 2 25 41 44 52 57 75 80 100
)

ROOT=$(realpath "$(dirname "$0")")
CONFIG_PATH=$(realpath "${1:-${ROOT}/configs/train.yaml}")
cd "${ROOT}/sound"

for OBJECT_ID in "${OBJECT_IDS[@]}"; do
    for DAMP_FLAG in --spatial-damping --no-spatial-damping; do
        echo "Training object ${OBJECT_ID} with ${CONFIG_PATH} ${DAMP_FLAG}"
        python trainer.py \
            --config "${CONFIG_PATH}" \
            --object-id "${OBJECT_ID}" \
            --ckpt-path "${ROOT}/results/feature_3dgs/${OBJECT_ID}/feature_3dgs.pt" \
            --sound-dataset-path "${ROOT}/sound/data/outputs/${OBJECT_ID}" \
            --result-dir "${ROOT}/results/{config_name}/{object_id}" \
            --run-contact-loc \
            --contact-export-topk 3 \
            "${DAMP_FLAG}"
    done
done
