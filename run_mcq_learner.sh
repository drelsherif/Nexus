#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_mcq_learner.sh — NEXUS MCQ Learner
#
# SQLite-backed pattern learning. 1000 cases/round, 16 rounds.
# LLM called for: (1) classification only, (2) MCQ generation per pattern.
# Pattern detection: pure SQL — zero LLM cost.
#
# Usage:
#   bash run_mcq_learner.sh          # fresh run
#   bash run_mcq_learner.sh warm     # warm restart
#   bash run_mcq_learner.sh 8        # fresh run, 8 rounds
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [[ -z "${AIHUB_API_KEY:-}" ]];      then echo "[ERROR] AIHUB_API_KEY not set"; exit 1; fi
if [[ -z "${AIHUB_AD_OBJECT_ID:-}" ]]; then echo "[ERROR] AIHUB_AD_OBJECT_ID not set"; exit 1; fi

CONFIG="task_configs/ade_cortex_v2.json"
OUT="run_mcq_learner"
ROUNDS=16
BATCH=1000
WORKERS=6
SEED=42
FRESH_FLAG="--fresh"
ARG="${1:-}"

if   [[ "${ARG}" == "warm" ]];           then FRESH_FLAG=""; echo "[MCQ] WARM RESTART"
elif [[ "${ARG}" =~ ^[0-9]+$ ]];         then ROUNDS="${ARG}"; echo "[MCQ] FRESH — ${ROUNDS} rounds"
else                                           echo "[MCQ] FRESH — ${ROUNDS} rounds × ${BATCH} cases"
fi

echo "[MCQ] Output: ${OUT}/ | Workers: ${WORKERS} | MinPatternErrors: 3"
echo ""

python3 -u nexus_mcq_learner.py \
  --config  "${CONFIG}" \
  --out     "${OUT}" \
  --rounds  "${ROUNDS}" \
  --batch-size "${BATCH}" \
  --min-pattern-errors 3 \
  --workers "${WORKERS}" \
  --seed    "${SEED}" \
  --ai-hub  \
  --ai-hub-key    "${AIHUB_API_KEY}" \
  --ai-hub-ad-id  "${AIHUB_AD_OBJECT_ID}" \
  ${FRESH_FLAG} \
  2>&1 | tee "${OUT}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "[MCQ] Done. DB: ${OUT}/nexus_memory.db | Log: ${OUT}/cortex_run_log.json"
