#!/bin/bash
set -euo pipefail

# Build one feature 3DGS per object.
# Usage:
#   bash run_feature_3dgs.sh
#   CUDA_VISIBLE_DEVICES=1 bash run_feature_3dgs.sh

OBJECT_IDS=(
    1 2 25 41 44 52 57 75 80 100
)

cd "$(dirname "$0")"

for OBJECT_ID in "${OBJECT_IDS[@]}"; do
    echo "Building feature 3DGS for object ${OBJECT_ID}"
    python vision/feature_3dgs.py --object-id "${OBJECT_ID}" "$@"
done
