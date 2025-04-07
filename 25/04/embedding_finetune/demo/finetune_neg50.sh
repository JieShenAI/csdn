torchrun --nproc_per_node=1 \
    -m FlagEmbedding.finetune.embedder.encoder_only.base \
    --model_name_or_path bert-base-uncased \
    --train_data ./ft_data/training_neg_50.json \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --output_dir ./output/bert-base-uncased_neg50 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --warmup_ratio 0.1 \
    --logging_steps 200 \
    --save_steps 2000 \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div

# --deepspeed config/ds_stage0.json \
# --gradient_checkpointing \
# CUDA_VISIBLE_DEVICES=0 TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \