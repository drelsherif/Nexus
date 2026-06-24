#!/bin/bash
mkdir -p run_09_rag
python3 -u run_rag.py \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_09_rag \
  --rounds 20 \
  --workers 4 \
  --k 5 \
  --threshold 5 \
  > run_09_rag/stdout.log 2>&1 &
echo "PID: $!"
tail -f run_09_rag/stdout.log
