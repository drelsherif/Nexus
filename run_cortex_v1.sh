#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_cortex_v1.sh — Enterprise run script for NEXUS Cortex v1.0
#
# NEXUS Cortex v1.0: Biologically-Grounded Adaptive Clinical NLP Classifier
# Implements: BCM plasticity, critical period dynamics, competitive routing,
#   route abstention, Jaccard overlap audit, MemoryTrace bequeathal
#
# Fixes all failure modes diagnosed in v3.04-enterprise:
#   FM-1 Routing dilution    → winner-take-all by specificity
#   FM-2 Route error voting  → abstention (None, excluded from ensemble)
#   FM-3 MCQ complexity harm → BCM-gated rehearsal weight
#   FM-4 Small eval pool     → 200 cases (was 100)
#   FM-5 Trigger overlap     → Jaccard audit before genesis (max=0.50)
#
# Prerequisites:
#   export AIHUB_API_KEY="..."
#   export AIHUB_AD_OBJECT_ID="..."
#
# Usage:
#   bash run_cortex_v1.sh         # fresh run (20 rounds × 250 cases)
#   bash run_cortex_v1.sh warm    # warm restart from existing cortex state
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Env validation ────────────────────────────────────────────────────────────
if [[ -z "${AIHUB_API_KEY:-}" ]]; then
  echo "[ERROR] AIHUB_API_KEY is not set."
  echo "  Run: export AIHUB_API_KEY=your_key_here"
  exit 1
fi

if [[ -z "${AIHUB_AD_OBJECT_ID:-}" ]]; then
  echo "[ERROR] AIHUB_AD_OBJECT_ID is not set."
  echo "  Run: export AIHUB_AD_OBJECT_ID=your_ad_object_id_here"
  exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
TASK="task_configs/ade_cortex_v1.json"
OUT="run_cortex_v1_enterprise"
ROUNDS=20
WORKERS=4
SEED=42

# ── Fresh vs warm start ────────────────────────────────────────────────────────
FRESH_FLAG="--fresh"
if [[ "${1:-}" == "warm" ]]; then
  FRESH_FLAG=""
  echo "[run_cortex_v1] WARM RESTART from ${OUT}/cortex_state.json"
else
  echo "[run_cortex_v1] FRESH START — ${ROUNDS} rounds × 250 cases"
fi

echo "[run_cortex_v1] Task:    ${TASK}"
echo "[run_cortex_v1] Output:  ${OUT}/"
echo "[run_cortex_v1] AI Hub:  ${AIHUB_API_KEY:0:8}..."
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python3 -u nexus_cortex_v1.py \
  --task "${TASK}" \
  --out "${OUT}" \
  --rounds "${ROUNDS}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --ai-hub \
  --ai-hub-key "${AIHUB_API_KEY}" \
  --ai-hub-ad-id "${AIHUB_AD_OBJECT_ID}" \
  ${FRESH_FLAG} \
  2>&1 | tee "${OUT}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "[run_cortex_v1] Run complete. Results in ${OUT}/cortex_run_log.json"
