#!/bin/bash
# run_v3_04.sh — NEXUS v3 warm restart
#
# Loads all learned state from run_v3_03:
#   - Tree structure + node state (MCQs, engrams, route weights, principles)
#   - nexus.db (baseline cache, eval history, MCQ dedup embeddings)
#   - Global RAG index (saves ~73s of embedding time)
#
# Then runs 20 fresh rounds with:
#   - Homeostatic controller active (principle rollback, node retirement, etc.)
#   - All v3_03 principles and MCQ teaching cases available from Round 1
#   - Threshold calibration starting from last known-good bias

set -e
mkdir -p run_v3_04

# ── Copy RAG index (saves embedding time) ────────────────────────────────────
if [ -d run_v3_03/global_rag_index ] && [ ! -d run_v3_04/global_rag_index ]; then
  echo "[setup] Copying RAG index from run_v3_03..."
  cp -r run_v3_03/global_rag_index run_v3_04/
fi

# ── Copy learned tree state (warm restart) ──────────────────────────────────
if [ -d run_v3_03/tree ] && [ ! -d run_v3_04/tree ]; then
  echo "[setup] Copying tree state from run_v3_03 (MCQs, engrams, weights, principles)..."
  cp -r run_v3_03/tree run_v3_04/
fi

# ── Copy nexus.db (baseline cache + MCQ embeddings for dedup) ───────────────
if [ -f run_v3_03/nexus.db ] && [ ! -f run_v3_04/nexus.db ]; then
  echo "[setup] Copying nexus.db from run_v3_03 (baseline cached, MCQ dedup embeddings)..."
  cp run_v3_03/nexus.db run_v3_04/
fi

echo "[setup] Warm restart ready. Starting v3_04..."

python3 -u nexus_v3.py \
  --task task_configs/ade_classification.json \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_v3_04 \
  --rounds 20 \
  > run_v3_04/stdout.log 2>&1 &

echo "PID: $!"
tail -f run_v3_04/stdout.log
