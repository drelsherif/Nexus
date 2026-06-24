#!/bin/bash
# run_parallel.sh — Run multiple NEXUS branches simultaneously.
#
# Launches N independent runs with different seeds/configs in parallel.
# Each run gets its own out-dir, drug registry, and nugget store.
# Results are compared at the end to find the best branch.
#
# Usage:
#   ./run_parallel.sh          # runs 3 branches (default)
#   ./run_parallel.sh 5        # runs 5 branches
#
# How it works:
#   Branch A: seed=42  (baseline config)
#   Branch B: seed=7   (different training sample order)
#   Branch C: seed=99  (different training sample order)
#   Branch D: seed=13
#   Branch E: seed=55
#
# All branches run rounds=10, batch=50, refine-probe=100.
# At completion, compare_runs.py prints the winner.

# ─── Configure ────────────────────────────────────────────────────────────────
GMAIL_ADDRESS="dryelsherif@gmail.com"
GMAIL_APP_PASS="mybl zfzo jmeg gerx"
PHONE="6464042406"
CARRIER="att"
ROUNDS=10
BATCH_SIZE=50
REFINE_PROBE=100
N_BRANCHES="${1:-3}"   # default 3, override with first argument

AI_HUB_KEY="${AIHUB_API_KEY}"
AI_HUB_AD="${AIHUB_AD_OBJECT_ID}"
# ──────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -z "$AI_HUB_KEY" ] || [ -z "$AI_HUB_AD" ]; then
    echo "[parallel] Error: AIHUB_API_KEY and AIHUB_AD_OBJECT_ID must be set."
    exit 1
fi

SEEDS=(42 7 99 13 55)
BRANCH_NAMES=(A B C D E)
PIDS=()
RUN_DIRS=()

echo "[parallel] Launching $N_BRANCHES parallel NEXUS branches..."
echo "[parallel] Rounds=$ROUNDS  Batch=$BATCH_SIZE  RefineProbe=$REFINE_PROBE"
echo ""

for i in $(seq 0 $((N_BRANCHES - 1))); do
    SEED="${SEEDS[$i]}"
    NAME="${BRANCH_NAMES[$i]}"
    OUT_DIR="run_05_branch_${NAME}"
    RUN_DIRS+=("$OUT_DIR")

    mkdir -p "$OUT_DIR"

    echo "[parallel] Starting Branch $NAME → $OUT_DIR (seed=$SEED)"

    "$PYTHON_BIN" nexus_run.py \
        --ai-hub \
        --ai-hub-key    "$AI_HUB_KEY" \
        --ai-hub-ad-id  "$AI_HUB_AD" \
        --rounds        "$ROUNDS" \
        --batch-size    "$BATCH_SIZE" \
        --refine-probe-size "$REFINE_PROBE" \
        --seed          "$SEED" \
        --out-dir       "$OUT_DIR" \
        --fresh-nuggets \
        --no-refine-root \
        > "$OUT_DIR/stdout.log" 2>&1 &

    PIDS+=($!)
    echo "[parallel]   PID ${PIDS[$i]}"
done

echo ""
echo "[parallel] All $N_BRANCHES branches running. PIDs: ${PIDS[*]}"
echo "[parallel] Monitor with:  tail -f run_05_branch_A/stdout.log"
echo "[parallel] To stop all:   kill ${PIDS[*]}"
echo ""

# Wait for all branches to finish
ALL_DONE=false
while ! $ALL_DONE; do
    sleep 60
    ALL_DONE=true
    RUNNING=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ALL_DONE=false
            RUNNING+=($pid)
        fi
    done
    if ! $ALL_DONE; then
        echo "[parallel] $(date '+%H:%M') — ${#RUNNING[@]}/$N_BRANCHES branches still running..."
    fi
done

echo ""
echo "[parallel] All branches complete. Comparing results..."
echo ""

# Compare results
"$PYTHON_BIN" compare_runs.py "${RUN_DIRS[@]}"

# Send SMS with winner
SUMMARY=$("$PYTHON_BIN" compare_runs.py "${RUN_DIRS[@]}" --short 2>/dev/null)
"$PYTHON_BIN" -c "
from notify import send_sms_email
send_sms_email(
    phone='$PHONE', carrier='$CARRIER',
    message='NEXUS parallel done.\n$SUMMARY',
    smtp_from='$GMAIL_ADDRESS', smtp_pass='$GMAIL_APP_PASS',
)
" 2>/dev/null

echo "[parallel] Done."
