# 初始化 Conda
eval "$(conda shell.bash hook)"

# 激活环境
conda activate factory


for i in {0..7}
do
    cd /mnt/mydisk/github/csdn/25/02/llm_industry_cls_infer/
    python generate_dataset.py

    cd /mnt/mydisk/github/csdn/25/02/llm_industry_cls_infer/infer
    python vllm_infer.py \
        --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
        --template qwen \
        --dataset_dir ../data \
        --dataset industry_cls \
        --top_p 0.7 \
        --temperature 0.95 \
        --cutoff_len 1024 \
        --max_new_tokens 2048 \
        --save_name output/qwen/generated_predictions_qwen_${i}.jsonl
done
