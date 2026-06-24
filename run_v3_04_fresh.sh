#!/bin/bash
# run_v3_04_fresh.sh — NEXUS v3_04 fully independent run
#
# Zero state copied from v3_03. Discovers its own MCQs, principles,
# engrams, and tree structure from scratch.
#
# The RAG index IS reused (saves ~73s) — it is just the corpus embeddings,
# not learned state. Every tree node, principle, MCQ, and engram is built
# fresh from Round 1.
#
# Purpose: validate that v3_04 independently discovers similar patterns
# to v3_03, which would confirm the architecture is stable and the learned
# features are genuine signal (not run-specific artifacts).

set -e
mkdir -p run_v3_04

# ── Optionally reuse RAG index (corpus embeddings only — not learned state) ──
# Comment out these lines to rebuild from scratch (adds ~73s to startup).
if [ -d run_v3_03/global_rag_index ] && [ ! -d run_v3_04/global_rag_index ]; then
  echo "[setup] Copying RAG index (corpus embeddings — not learned state)..."
  cp -r run_v3_03/global_rag_index run_v3_04/
fi

echo "[setup] Fresh start — no tree, no principles, no MCQs, no DB from v3_03."
echo "[setup] Starting v3_04 independent run..."

python3 -u nexus_v3.py \
  --task task_configs/ade_classification.json \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_v3_04 \
  --rounds 20 \
  --fresh \
  > run_v3_04/stdout.log 2>&1 &

echo "PID: $!"
tail -f run_v3_04/stdout.log
