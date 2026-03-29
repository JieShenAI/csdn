#!/bin/bash
export SGLANG_DISABLE_CUDNN_CHECK=1

/home/jie/local_projects/sglang_serve/.venv/bin/python -m sglang.launch_server \
--model-path Qwen/Qwen3.5-9B \
--served-model-name qwen3.5-9b \
--host 0.0.0.0 \
--port 8000 \
--tp-size 1 \
--mem-fraction-static 0.80 \
--kv-cache-dtype fp8_e4m3 \
--context-length 131072 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--attention-backend triton \
--chunked-prefill-size 65535 \
--triton-attention-num-kv-splits 4 \
--max-running-requests 4 \
--enable-tokenizer-batch-encode \
--disable-radix-cache \
--enable-metrics