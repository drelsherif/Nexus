#!/bin/bash
# run_nexus.sh — NEXUS experiment launcher
#
# Usage:
#   chmod +x run_nexus.sh
#   ./run_nexus.sh
#
# Edit the variables below once, then just run this script.

# ─── Configure once ───────────────────────────────────────────────────────────
GMAIL_ADDRESS="dryelsherif@gmail.com"
GMAIL_APP_PASS="mybl zfzo jmeg gerx"
PHONE="6464042406"
CARRIER="att"
OUT_DIR="run_05"
ROUNDS=10
BATCH_SIZE=50        # increased from 20 — more signal per round
REFINE_PROBE=100

# AI Hub credentials (or export these in your shell profile)
AI_HUB_KEY="${AIHUB_API_KEY}"
AI_HUB_AD="${AIHUB_AD_OBJECT_ID}"
# ──────────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[run_nexus] Error: '$PYTHON_BIN' not found."
    exit 1
fi

if [ -z "$AI_HUB_KEY" ] || [ -z "$AI_HUB_AD" ]; then
    echo "[run_nexus] Error: AIHUB_API_KEY and AIHUB_AD_OBJECT_ID must be set."
    exit 1
fi

mkdir -p "$OUT_DIR"

# Start SMS listener in background
echo "[run_nexus] Starting SMS listener..."
"$PYTHON_BIN" sms_listener.py \
    --email   "$GMAIL_ADDRESS" \
    --pass    "$GMAIL_APP_PASS" \
    --phone   "$PHONE" \
    --carrier "$CARRIER" \
    --sentinel-dir "$OUT_DIR" &
LISTENER_PID=$!
echo "[run_nexus] Listener PID: $LISTENER_PID"
echo "[run_nexus] Text $PHONE these commands from your phone to control the run:"
echo "   KILL   → stop after current round"
echo "   PAUSE  → pause after current round"
echo "   RESUME → resume a paused run"
echo "   STATUS → get current round + F1"
echo ""

# Run NEXUS — run_05 improvements:
#   --no-refine-root  ROOT is at its prompt ceiling, skip it
#   --batch-size 50   more signal per round
#   --fresh-nuggets   clean start
"$PYTHON_BIN" nexus_run.py \
    --ai-hub \
    --ai-hub-key    "$AI_HUB_KEY" \
    --ai-hub-ad-id  "$AI_HUB_AD" \
    --rounds        "$ROUNDS" \
    --batch-size    "$BATCH_SIZE" \
    --refine-probe-size "$REFINE_PROBE" \
    --out-dir       "$OUT_DIR" \
    --fresh-nuggets \
    --no-refine-root \
    --notify-phone  "$PHONE" \
    --notify-carrier "$CARRIER" \
    --notify-email  "$GMAIL_ADDRESS" \
    --notify-pass   "$GMAIL_APP_PASS"

echo "[run_nexus] Run complete. Stopping SMS listener (PID $LISTENER_PID)..."
kill $LISTENER_PID 2>/dev/null
echo "[run_nexus] Done."
