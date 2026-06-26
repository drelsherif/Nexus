#!/bin/bash
# NEXUS v3 — Held-out inference run
# Evaluates the trained v3.04 tree on ~300 cases never seen during training.
# No training, no synthesis — only classification calls (~4 LLM calls/case).
set -e

python3 -u infer_v3.py \
    --run run_v3_04 \
    --ai-hub \
    --ai-hub-key "$AIHUB_API_KEY" \
    --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
    --n-cases 300 \
    --tail-size 5000 \
    --ade-bias 1.0 \
    --workers 4 \
    2>&1 | tee run_v3_04/infer_stdout.log
