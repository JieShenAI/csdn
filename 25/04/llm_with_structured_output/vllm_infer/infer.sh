python vllm_infer.py \
            --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
            --template qwen \
            --dataset "ag_news_test" \
            --dataset_dir data \
            --save_name output/vllm_ag_news_test.json \
            > logs/vllm_ag_news_test.log 2>&1