# 初始化 Conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate factory


for i in {7..7}
do
    cd /mnt/mydisk/github/csdn/25/02/llm_industry_cls_infer/
    python generate_dataset.py

    cd /mnt/mydisk/github/csdn/25/02/llm_industry_cls_infer/infer
    python vllm_infer.py \
        --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --template qwen \
        --template deepseek3 \
        --dataset_dir ../data \
        --dataset industry_cls \
        --top_p 0.7 \
        --temperature 0.95 \
        --cutoff_len 1024 \
        --max_new_tokens 2048 \
        --save_name output/deepseek/generated_predictions_deepseek3_${i}.jsonl
done
