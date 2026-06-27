#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_apex.sh — NEXUS Apex Learner
#
# Architecture: Predictive-Error + Contrastive Pairs + Consolidation
# 1000 cases/round × 16 rounds | Two-pass Gamma-Theta | Sleep every 3R
#
# Usage:
#   bash run_apex.sh          # fresh 16-round run
#   bash run_apex.sh warm     # warm restart from last checkpoint
#   bash run_apex.sh 8        # fresh run, 8 rounds
#   bash run_apex.sh mock     # mock mode (no API calls, testing only)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ARG="${1:-}"

# Mock mode bypass (no API keys needed)
if [[ "${ARG}" == "mock" ]]; then
  echo "[Apex] MOCK MODE — no API calls"
  python3 -u nexus_apex.py \
    --config  "task_configs/ade_cortex_v2.json" \
    --out     "run_apex_mock" \
    --rounds  4 \
    --batch-size 50 \
    --workers 2 \
    --seed    42 \
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
ROUNDS=16
BATCH=1000
WORKERS=6
SEED=42
FRESH_FLAG="--fresh"

if   [[ "${ARG}" == "warm" ]];         then FRESH_FLAG=""; echo "[Apex] WARM RESTART"
elif [[ "${ARG}" =~ ^[0-9]+$ ]];       then ROUNDS="${ARG}"; echo "[Apex] FRESH — ${ROUNDS} rounds"
else                                         echo "[Apex] FRESH — ${ROUNDS} rounds × ${BATCH} cases"
fi

echo "[Apex] Output: ${OUT}/ | Workers: ${WORKERS} | Consolidation every 3 rounds"
echo "[Apex] Architecture: Gamma-Theta | Contrastive Pairs | EMA Threshold | ACh Gating"
echo ""

python3 -u nexus_apex.py \
  --config       "${CONFIG}" \
  --out          "${OUT}" \
  --rounds       "${ROUNDS}" \
  --batch-size   "${BATCH}" \
  --workers      "${WORKERS}" \
  --seed         "${SEED}" \
  --ai-hub \
  --ai-hub-key      "${AIHUB_API_KEY}" \
  --ai-hub-ad-id    "${AIHUB_AD_OBJECT_ID}" \
  ${FRESH_FLAG} \
  2>&1 | tee "${OUT}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "[Apex] Done. DB: ${OUT}/nexus_apex.db"
