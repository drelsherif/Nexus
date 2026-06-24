#!/bin/bash
# NEXUS v3 — Public run script
# Works with any OpenAI-compatible API: OpenAI, Anthropic, Ollama, Together, etc.
#
# Usage:
#   OpenAI:   export OPENAI_API_KEY=sk-...  then ./run_v3_public.sh
#   Ollama:   ./run_v3_public.sh --base-url http://localhost:11434/v1 --classify llama3.2 --synth llama3.1:70b
#   Anthropic via openai-compat: export OPENAI_API_KEY=sk-ant-...
#             ./run_v3_public.sh --base-url https://api.anthropic.com/v1 --classify claude-haiku-4-5 --synth claude-sonnet-4-5

set -e

# Defaults (override with flags below)
CLASSIFY_MODEL="gpt-4o-mini"
SYNTH_MODEL="gpt-4o"
BASE_URL=""
OUT_DIR="run_v3_01"

# Parse optional flags
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --classify)  CLASSIFY_MODEL="$2"; shift 2 ;;
        --synth)     SYNTH_MODEL="$2"; shift 2 ;;
        --out)       OUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUT_DIR"

BASE_URL_ARG=""
if [ -n "$BASE_URL" ]; then
    BASE_URL_ARG="--openai-base-url $BASE_URL"
fi

echo "Starting NEXUS v3"
echo "  classify model : $CLASSIFY_MODEL"
echo "  synth model    : $SYNTH_MODEL"
echo "  output dir     : $OUT_DIR"
[ -n "$BASE_URL" ] && echo "  base URL       : $BASE_URL"
echo ""

python3 -u nexus_v3.py \
    --task task_configs/ade_classification.json \
    --openai \
    --openai-classify-model "$CLASSIFY_MODEL" \
    --openai-synth-model    "$SYNTH_MODEL" \
    $BASE_URL_ARG \
    --out "$OUT_DIR" --rounds 20 --fresh \
    2>&1 | tee "$OUT_DIR/stdout.log"
