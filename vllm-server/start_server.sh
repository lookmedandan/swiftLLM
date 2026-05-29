#!/bin/bash

MODEL_PATH="/root/AI_agent/models/qwen1.5b_muice_dpo_final"
LOG_FILE="/root/AI_agent/logs/vllm-$(date +%Y%m%d-%H%M%S).log"

cd /root/AI_agent/vllm-server

python -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --served-model-name qwen-muice-dpo \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.8 \
  >> $LOG_FILE 2>&1 &

echo "vLLM 服务已启动，PID: $!"
echo "日志: $LOG_FILE"