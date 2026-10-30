#!/bin/bash
set -euo pipefail

# Generate impact_sound.json + partition.json for ObjectFolder-Real objects.
#
# Usage:
#   bash sound/data/impact_sound_data.sh                # OBJECT_IDS
#   bash sound/data/impact_sound_data.sh 80             # single id
#   bash sound/data/impact_sound_data.sh 1 100          # range [1, 100]
#   bash sound/data/impact_sound_data.sh 5 12 47 80     # explicit list

OBJECT_IDS=(
    1 2 25 41 44 52 57 75 80 100
)

if (( $# == 2 )) && [[ "$1" =~ ^[0-9]+$ ]] && [[ "$2" =~ ^[0-9]+$ ]] && (( $1 <= $2 )); then
    mapfile -t OBJECT_IDS < <(seq "$1" "$2")
elif (( $# > 0 )); then
    OBJECT_IDS=("$@")
fi

cd "$(dirname "$0")/../.."

python sound/data/impact_sound_data.py \
    --output-dir sound/data/outputs \
    --object-ids "${OBJECT_IDS[@]}"
