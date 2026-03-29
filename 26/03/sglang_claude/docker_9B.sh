docker run -it --rm \
  --gpus all \
  --net=host \
  -p 8000:8000 \
  -v /home/jie/.cache/huggingface:/root/.cache/huggingface \
  -e SGLANG_DISABLE_CUDNN_CHECK=1 \
  -e HF_ENDPOINT="https://hf-mirror.com" \
  sglang-jie:latest \
  python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --served-model-name qwen3.5-9b \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen \
  --attention-backend triton \
  --chunked-prefill-size 4096 \
  --dtype bfloat16 \
  --enable-tokenizer-batch-encode \
  --enable-metrics 
