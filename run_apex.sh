#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_apex.sh — NEXUS Apex Learner v2 (Curriculum Edition)
#
# Architecture: Predictive-Error + Contrastive Pairs + Consolidation
#               + Salience-Gated Memory + Developmental Curriculum
#
# Curriculum schedule:
#   Phase 1 (R01-R10):  50 cases/round  — high error rate, fast pair generation
#   Phase 2 (R11-R20): 100 cases/round  — consolidation of early lessons
#   Phase 3 (R21-R30): 250 cases/round  — generalisation stress test
#   Phase 4 (R31-R40): 500 cases/round  — full complexity fine-tuning
#
# Usage:
#   bash run_apex.sh              # fresh curriculum run (40 rounds, 4 phases)
#   bash run_apex.sh warm         # warm restart from last checkpoint
#   bash run_apex.sh fixed 16     # fresh fixed-batch run (1000/round × 16R)
#   bash run_apex.sh mock         # mock mode (no API calls, testing only)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ARG="${1:-}"
ARG2="${2:-}"

# Mock mode bypass (no API keys needed)
if [[ "${ARG}" == "mock" ]]; then
  echo "[Apex] MOCK MODE — no API calls"
  python3 -u nexus_apex.py \
    --config     "task_configs/ade_cortex_v2.json" \
    --out        "run_apex_mock" \
    --rounds     4 \
    --batch-size 50 \
    --workers    2 \
    --seed       42 \
    --fresh \
    --mock \
    2>&1 | tee "run_apex_mock_$(date +%Y%m%d_%H%M%S).log"
  exit 0
fi

# Real runs require API credentials
if [[ -z "${AIHUB_API_KEY:-}" ]];       then echo "[ERROR] AIHUB_API_KEY not set";      exit 1; fi
if [[ -z "${AIHUB_AD_OBJECT_ID:-}" ]];  then echo "[ERROR] AIHUB_AD_OBJECT_ID not set"; exit 1; fi

CONFIG="task_configs/ade_cortex_v2.json"
OUT="run_apex"
WORKERS=6
SEED=42

if [[ "${ARG}" == "warm" ]]; then
  # ── Warm restart: resume curriculum from checkpoint ─────────────────────
  echo "[Apex] WARM RESTART — resuming curriculum"
  python3 -u nexus_apex.py \
    --config     "${CONFIG}" \
    --out        "${OUT}" \
    --curriculum \
    --workers    "${WORKERS}" \
    --seed       "${SEED}" \
    --ai-hub \
    --ai-hub-key      "${AIHUB_API_KEY}" \
    --ai-hub-ad-id    "${AIHUB_AD_OBJECT_ID}" \
    2>&1 | tee "${OUT}_warm_$(date +%Y%m%d_%H%M%S).log"

elif [[ "${ARG}" == "fixed" ]]; then
  # ── Fixed batch mode: legacy style ─────────────────────────────────────
  ROUNDS="${ARG2:-16}"
  BATCH=1000
  echo "[Apex] FIXED MODE — ${ROUNDS} rounds × ${BATCH} cases"
  python3 -u nexus_apex.py \
    --config     "${CONFIG}" \
    --out        "${OUT}" \
    --rounds     "${ROUNDS}" \
    --batch-size "${BATCH}" \
    --workers    "${WORKERS}" \
    --seed       "${SEED}" \
    --ai-hub \
    --ai-hub-key      "${AIHUB_API_KEY}" \
    --ai-hub-ad-id    "${AIHUB_AD_OBJECT_ID}" \
    --fresh \
    2>&1 | tee "${OUT}_fixed_$(date +%Y%m%d_%H%M%S).log"

else
  # ── Default: fresh curriculum run ───────────────────────────────────────
  echo "[Apex] CURRICULUM RUN — 40 rounds, 4 phases (50→100→250→500 cases)"
  echo "[Apex] Output: ${OUT}/ | Workers: ${WORKERS}"
  echo "[Apex] Architecture: Gamma-Theta | Contrastive Pairs | Salience-Gated Memory"
  echo "[Apex] Consolidation every 3 rounds with 1-round cooldown"
  echo ""
  python3 -u nexus_apex.py \
    --config     "${CONFIG}" \
    --out        "${OUT}" \
    --curriculum \
    --workers    "${WORKERS}" \
    --seed       "${SEED}" \
    --ai-hub \
    --ai-hub-key      "${AIHUB_API_KEY}" \
    --ai-hub-ad-id    "${AIHUB_AD_OBJECT_ID}" \
    --fresh \
    2>&1 | tee "${OUT}_$(date +%Y%m%d_%H%M%S).log"
fi

echo ""
echo "[Apex] Done. DB: ${OUT}/nexus_apex.db"
