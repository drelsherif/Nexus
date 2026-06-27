#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_cortex_v2.sh — Enterprise run script for NEXUS Cortex v2.0
#
# NEXUS Cortex v2.0: Guided Developmental Cortical NLP Classifier
# New in v2 (added to all v1 mechanisms):
#   MCQLibrary   — contrastive MCQ lessons replace raw error buffer
#   RejectedProposalMemory — persistent genesis failure log fed to LLM
#   MetaAgent    — LLM as diagnostic physician (fires on F1 decline)
#   Shadow column period — 1-round warm-up before routing activation
#   Trigger-scoped genesis probe — evaluate only trigger-matched cases
#
# Retained from v1.0:
#   BCM plasticity, critical period, competitive routing, homeostasis,
#   neurogenesis + apoptosis, EnggramClusters, MemoryTrace bequeathal,
#   FM-2/5/6/7/8 fixes, warm restart
#
# Prerequisites:
#   export AIHUB_API_KEY="..."
#   export AIHUB_AD_OBJECT_ID="..."
#
# Usage:
#   bash run_cortex_v2.sh         # fresh run (20 rounds × 250 cases)
#   bash run_cortex_v2.sh warm    # warm restart from existing cortex state
#   bash run_cortex_v2.sh 10      # fresh run, custom round count
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
TASK="task_configs/ade_cortex_v2.json"
OUT="run_cortex_v2_enterprise"
ROUNDS=20
WORKERS=4
SEED=42

# ── Argument parsing ──────────────────────────────────────────────────────────
FRESH_FLAG="--fresh"
ARG="${1:-}"

if [[ "${ARG}" == "warm" ]]; then
  FRESH_FLAG=""
  echo "[run_cortex_v2] WARM RESTART from ${OUT}/cortex_state.json"
elif [[ "${ARG}" =~ ^[0-9]+$ ]]; then
  ROUNDS="${ARG}"
  echo "[run_cortex_v2] FRESH START — ${ROUNDS} rounds × 250 cases"
else
  echo "[run_cortex_v2] FRESH START — ${ROUNDS} rounds × 250 cases"
fi

echo "[run_cortex_v2] Version:  2.0"
echo "[run_cortex_v2] Task:     ${TASK}"
echo "[run_cortex_v2] Output:   ${OUT}/"
echo "[run_cortex_v2] AI Hub:   ${AIHUB_API_KEY:0:8}..."
echo "[run_cortex_v2] New:      MCQLibrary + RejectedProposalMemory + MetaAgent + Shadow Columns"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
python3 -u nexus_cortex_v2.py \
  --config "${TASK}" \
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
echo "[run_cortex_v2] Run complete."
echo "[run_cortex_v2] State:    ${OUT}/cortex_state.json"
echo "[run_cortex_v2] Log:      ${OUT}/cortex_run_log.json"
echo "[run_cortex_v2] Rejected: ${OUT}/rejected_proposals.json"
