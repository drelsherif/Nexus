#!/bin/bash
# run_v3_05.sh — NEXUS v3.05 — Fresh start (build from scratch)
#
# What's new in v3.05 (MCQ system upgrade):
#   - Near-miss MCQs: correct-but-close predictions (margin < 20%)
#   - Positive anchor MCQs: high-confidence correct (≥95% agreement, ≥90% conf)
#   - Difficulty-weighted retrieval (hard=1.5x, near_miss=1.3x, error=1.2x)
#   - Cross-node global MCQ pool for thin libraries (<3 MCQs)
#   - Positive anchors EXCLUDED from retrieval context (prevents NOT_ADE bias)
#   - Case tracking: trained_cases.jsonl + unseen_cases.jsonl saved at end
#   - Route error logging (stderr) for silent failure detection
#   - Startup API sanity check — fails fast if route calls don't work
#   - Dynamic bias sweep (0.3–4.0 with adaptive fine-grained steps)
#   - Separated ADE/NOT_ADE RAG examples in route prompts
#
# Run parameters:
#   - 20 rounds × 250 batch cases = up to 5,000 training exposures
#   - Fresh tree — no warm restart from v3.04
#   - Eval: fixed 100-case pool (locked, never trained on)
#   - Output: run_v3_05/
#
# REQUIRED environment variables:
#   export AIHUB_API_KEY='your-key-here'
#   export AIHUB_AD_OBJECT_ID='your-uuid-here'

set -e

# ── Environment variable validation ─────────────────────────────────────────
if [ -z "${AIHUB_API_KEY:-}" ]; then
  echo ""
  echo "❌  ERROR: AIHUB_API_KEY is not set."
  echo ""
  echo "    Set it before running:"
  echo "      export AIHUB_API_KEY='your-api-key-here'"
  echo ""
  exit 1
fi

if [ -z "${AIHUB_AD_OBJECT_ID:-}" ]; then
  echo ""
  echo "❌  ERROR: AIHUB_AD_OBJECT_ID is not set."
  echo ""
  echo "    Set it before running:"
  echo "      export AIHUB_AD_OBJECT_ID='your-ad-object-uuid-here'"
  echo ""
  exit 1
fi

echo "✓  AIHUB_API_KEY set (${#AIHUB_API_KEY} chars)"
echo "✓  AIHUB_AD_OBJECT_ID set (${AIHUB_AD_OBJECT_ID:0:8}...)"
echo ""

mkdir -p run_v3_05

# ── Safety check: don't overwrite a completed run ──────────────────────────
if [ -f run_v3_05/v3_summary.json ]; then
  echo "❌  run_v3_05/v3_summary.json already exists — this run already completed."
  echo "    To re-run from scratch:"
  echo "      rm -rf run_v3_05"
  echo "      bash run_v3_05.sh"
  exit 1
fi

echo "[setup] Fresh start — building v3.05 tree from seed nodes..."
echo "[setup] 20 rounds × 250 batch cases, case tracking enabled"
echo "[setup] Startup API test will run before training begins"
echo ""

python3 -u nexus_v3.py \
  --task task_configs/ade_classification.json \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_v3_05 \
  --rounds 20 \
  --fresh \
  > run_v3_05/stdout.log 2>&1 &

echo "PID: $!"
echo "Log: run_v3_05/stdout.log"
echo ""
tail -f run_v3_05/stdout.log
