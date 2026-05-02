#!/bin/bash
# Run training on a single dataset
# Usage: bash scripts/run_single.sh <dataset> [gpu]
# Example: bash scripts/run_single.sh last-fm 0
set -e

DATASET="${1:?Usage: bash scripts/run_single.sh <dataset> [gpu]}"
GPU="${2:-0}"

DATA_PATH="data/${DATASET}/"
CONFIG_PATH="configs/${DATASET}.yaml"

if [ ! -d "$DATA_PATH" ]; then
    echo "Error: dataset directory not found: $DATA_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: config file not found: $CONFIG_PATH"
    exit 1
fi

if [[ "$DATASET" == "amazon-book" || "$DATASET" == "new_amazon-book" ]]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

echo "=== Training on ${DATASET} (GPU ${GPU}) ==="
python train.py --data_path "$DATA_PATH" --config "$CONFIG_PATH" --gpu "$GPU"
