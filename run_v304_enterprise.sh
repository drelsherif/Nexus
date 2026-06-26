#!/bin/bash
# run_v304_enterprise.sh — NEXUS v3.04 config on enterprise AIHub
#
# Uses the v3.04 task config (200-case batches, simple MCQ system — no
# near-miss/positive anchor upgrades) on the current fixed codebase.
#
# Why this exists:
#   v3.04 achieved F1=0.9455 via warm restart from v3.03 state.
#   This run replicates the v3.04 architecture (simpler MCQ, 200 batch)
#   from a fresh start, with all v3.05 stability fixes applied:
#     - Separated ADE/NOT_ADE RAG examples in route prompts
#     - _safe_float (no confidence string crashes)
#     - Route error logging (stderr)
#     - Startup API sanity check
#     - Dynamic bias sweep (0.3–4.0)
#     - Local corpus cache (no HuggingFace dependency)
#
# Run parameters:
#   - 20 rounds × 200 batch cases
#   - Fresh tree — no warm restart
#   - Eval: fixed 100-case pool
#   - Output: run_v304_enterprise/

set -e

# ── Env var validation ────────────────────────────────────────────────────────
if [ -z "${AIHUB_API_KEY:-}" ]; then
  echo ""
  echo "❌  ERROR: AIHUB_API_KEY is not set."
  echo "    export AIHUB_API_KEY='your-key-here'"
  echo ""
  exit 1
fi

if [ -z "${AIHUB_AD_OBJECT_ID:-}" ]; then
  echo ""
  echo "❌  ERROR: AIHUB_AD_OBJECT_ID is not set."
  echo "    export AIHUB_AD_OBJECT_ID='your-uuid-here'"
  echo ""
  exit 1
fi

echo "✓  AIHUB_API_KEY set (${#AIHUB_API_KEY} chars)"
echo "✓  AIHUB_AD_OBJECT_ID set (${AIHUB_AD_OBJECT_ID:0:8}...)"
echo ""

mkdir -p run_v304_enterprise

# ── Safety check ─────────────────────────────────────────────────────────────
if [ -f run_v304_enterprise/v3_summary.json ]; then
  echo "❌  run_v304_enterprise/ already completed."
  echo "    To re-run: rm -rf run_v304_enterprise && bash run_v304_enterprise.sh"
  exit 1
fi

# ── Reuse RAG index if available (saves ~75s) ─────────────────────────────────
if [ -d run_v3_05/global_rag_index ] && [ ! -d run_v304_enterprise/global_rag_index ]; then
  echo "[setup] Copying RAG index from run_v3_05 (corpus embeddings only — not learned state)..."
  cp -r run_v3_05/global_rag_index run_v304_enterprise/
elif [ -d run_v3_05_mock/global_rag_index ] && [ ! -d run_v304_enterprise/global_rag_index ]; then
  echo "[setup] Copying RAG index from run_v3_05_mock..."
  cp -r run_v3_05_mock/global_rag_index run_v304_enterprise/
fi

echo "[setup] Fresh start — v3.04 config (200-case batches, simple MCQ)"
echo "[setup] 20 rounds × 200 batch cases"
echo "[setup] Startup API test will run before training begins"
echo ""

python3 -u nexus_v3.py \
  --task task_configs/ade_classification_v304.json \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_v304_enterprise \
  --rounds 20 \
  --fresh \
  > run_v304_enterprise/stdout.log 2>&1 &

echo "PID: $!"
echo "Log: run_v304_enterprise/stdout.log"
echo ""
tail -f run_v304_enterprise/stdout.log
