#!/bin/bash
mkdir -p run_v3_03
# Reuse the RAG index from v3_02 — saves 73s of embedding time
if [ -d run_v3_02/global_rag_index ] && [ ! -d run_v3_03/global_rag_index ]; then
  echo "[setup] Copying RAG index from run_v3_02..."
  cp -r run_v3_02/global_rag_index run_v3_03/
fi
python3 -u nexus_v3.py \
  --task task_configs/ade_classification.json \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_v3_03 \
  --rounds 20 \
  > run_v3_03/stdout.log 2>&1 &
echo "PID: $!"
tail -f run_v3_03/stdout.log
