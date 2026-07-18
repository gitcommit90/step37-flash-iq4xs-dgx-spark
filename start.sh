#!/usr/bin/env bash
# Step-3.7-Flash IQ4_XS on DGX Spark (GB10) — 256K single-stream serve.
# Requires: llama.cpp built from stepfun-ai/llama.cpp branch step3.7,
#           IQ4_XS shards downloaded (see README).
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp-step37/build/bin/llama-server}"
MODEL_DIR="${MODEL_DIR:-$HOME/llm/step37-flash-iq4xs}"
MODEL="$MODEL_DIR/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"
PORT="${PORT:-8088}"
CTX="${CTX:-262144}"
PARALLEL="${PARALLEL:-1}"
LOG="$MODEL_DIR/server-256k.log"

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found at $LLAMA_BIN — build it first (see README step 1)." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model shard not found at $MODEL — download it first (see README step 2)." >&2
  exit 1
fi

if pgrep -f "llama-server.*Step-3.7-flash" >/dev/null; then
  echo "A Step-3.7 llama-server already appears to be running. ./stop.sh first." >&2
  exit 1
fi

echo "Starting Step-3.7-Flash IQ4_XS on port $PORT (ctx=$CTX, parallel=$PARALLEL) — 256K goal"
echo "Log: $LOG  (first load takes ~6-8 min; API answers 503 until ready)"
nohup "$LLAMA_BIN" \
  --model "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --ctx-size "$CTX" --parallel "$PARALLEL" \
  --batch-size 2048 --ubatch-size 1024 \
  --n-gpu-layers 99 --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --jinja --metrics --no-webui \
  > "$LOG" 2>&1 &
echo "PID: $!"

for i in $(seq 1 144); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Server is UP: http://127.0.0.1:$PORT (n_ctx=$CTX, parallel=$PARALLEL)"
    exit 0
  fi
  sleep 5
done
echo "Server did not become healthy in 12 min — check $LOG" >&2
exit 1
