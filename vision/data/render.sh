#!/bin/bash
#
# Usage:
#   bash vision/data/render.sh                # render OBJECT_IDS below
#   bash vision/data/render.sh 5-20           # render ids 5..20 (inclusive)
#   bash vision/data/render.sh 1 2 25 41 44 52 57 75 80 100      # render the listed ids
#
# Already-rendered objects (those with both transforms.json and mesh.ply
# under vision/data/outputs/<id>/) are skipped so the script can be re-run.

set -euo pipefail

OBJECT_IDS=(
    1 2 25 41 44 52 57 75 80 100
)

cd "$(dirname "$0")/../.."
LOG_DIR=logs/render
mkdir -p "${LOG_DIR}"

# Parse arguments -> OBJECT_IDS
if (( $# == 1 )) && [[ "$1" =~ ^[0-9]+-[0-9]+$ ]]; then
    A=${1%-*}
    B=${1#*-}
    if (( A > B )); then
        echo "Error: range '$1' invalid (start > end)." >&2
        exit 2
    fi
    OBJECT_IDS=( $(seq "$A" "$B") )
elif (( $# > 0 )); then
    for x in "$@"; do
        if ! [[ "$x" =~ ^[0-9]+$ ]]; then
            echo "Error: invalid object id '$x' (expected positive integer or 'a-b' range)." >&2
            exit 2
        fi
    done
    OBJECT_IDS=( "$@" )
fi

# Run renders sequentially
echo "Rendering ${#OBJECT_IDS[@]} objects locally: ${OBJECT_IDS[*]}"

for OBJECT_ID in "${OBJECT_IDS[@]}"; do
    OUT_DIR="vision/data/outputs/${OBJECT_ID}"
    if [[ -f "${OUT_DIR}/transforms.json" && -f "${OUT_DIR}/mesh.ply" ]]; then
        echo "[render] object ${OBJECT_ID} already rendered, skipping."
        continue
    fi

    echo "[render] === object ${OBJECT_ID} ==="
    if ! python vision/data/render.py "${OBJECT_ID}" 2>&1 \
            | tee "${LOG_DIR}/render_${OBJECT_ID}.log"; then
        echo "[render] FAILED on object ${OBJECT_ID}."
        exit 1
    fi
done

echo "All renders done."
