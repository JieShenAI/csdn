# 对所有切分后的数据集进行训练
cd LLaMA-Factory
data_files=(llm_train_100 llm_train_500 llm_train_1000 llm_train_2000)
echo ${data_files[@]}

for data_file in ${data_files[@]}; do
    echo ${data_file}
    llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path ZhipuAI/glm-4-9b-chat \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --template glm4 \
        --flash_attn auto \
        --dataset_dir data \
        --dataset ${data_file} \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --num_train_epochs 3.0 \
        --max_samples 100000 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_steps 100 \
        --warmup_steps 0 \
        --optim adamw_torch \
        --packing False \
        --report_to none \
        --output_dir saves/GLM-4-9B-Chat/lora/240731-${data_file} \
        --fp16 True \
        --plot_loss True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0 \
        --lora_target all
done

# nohup bash train.sh > train.log 2>&1 &