# 初始化 Conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate factory

cutoff_len=20
max_new_tokens=100
python vllm_infer.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --template deepseek3 \
    --dataset_dir ../data \
    --dataset long_query \
    --top_p 0.7 \
    --temperature 0.95 \
    --max_samples 5 \
    --cutoff_len $cutoff_len \
    --max_new_tokens $max_new_tokens \
    --save_name output/generated_predictions_${cutoff_len}_${max_new_tokens}.jsonl
    


cutoff_len=2048
max_new_tokens=4096
python vllm_infer.py \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --template deepseek3 \
    --dataset_dir ../data \
    --dataset long_query \
    --top_p 0.7 \
    --temperature 0.95 \
    --cutoff_len $cutoff_len \
    --max_new_tokens $max_new_tokens \
    --save_name output/generated_predictions_${cutoff_len}_${max_new_tokens}.jsonl