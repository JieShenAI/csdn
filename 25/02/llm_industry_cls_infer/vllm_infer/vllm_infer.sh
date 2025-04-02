python vllm_infer.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --template deepseek3 \
    --dataset_dir ../data \
    --dataset long_query \
    --top_p 0.7 \
    --temperature 0.95 \
    --cutoff_len 2048 \
    --max_new_tokens 4096 \
    --save_name output/generated_predictions_vllm_infer.jsonl