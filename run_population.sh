#!/bin/bash
# run_population.sh
# Runs population-based NEXUS branches.

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[run_population] Error: '$PYTHON_BIN' not found."
    exit 1
fi

if [ -z "$AIHUB_API_KEY" ] || [ -z "$AIHUB_AD_OBJECT_ID" ]; then
    echo "[run_population] Error: AIHUB_API_KEY and AIHUB_AD_OBJECT_ID must be exported before running."
    exit 1
fi

"$PYTHON_BIN" nexus_population.py \
    --ai-hub \
    --out-dir population_runs \
    --branches 10 \
    --max-parallel 3 \
    --batch-start 5 \
    --batch-end 10 \
    --rounds 10 \
    --probe-size 300 \
    --eval-size 200 \
    --refine-probe-size 30
