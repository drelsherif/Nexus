#!/bin/bash
mkdir -p run_v3_02
python3 -u nexus_v3.py \
  --task task_configs/ade_classification.json \
  --ai-hub \
  --ai-hub-key "$AIHUB_API_KEY" \
  --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \
  --out run_v3_02 \
  --rounds 20 \
  > run_v3_02/stdout.log 2>&1 &
echo "PID: $!"
tail -f run_v3_02/stdout.log
