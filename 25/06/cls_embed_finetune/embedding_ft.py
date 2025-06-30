"""
reference: 
torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path BAAI/bge-large-en-v1.5 \
    --cache_dir ./cache/model \
    --train_data ./example_data/retrieval \
    			 ./example_data/sts/sts.jsonl \
    			 ./example_data/classification-no_in_batch_neg \
    			 ./example_data/clustering-no_in_batch_neg \
    --cache_path ./cache/data \
    --train_group_size 8 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}{}' \
    --knowledge_distillation False \
	--output_dir ./test_encoder_only_base_bge-large-en-v1.5 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ../ds_stage0.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type kl_div
"""

from transformers import HfArgumentParser

from FlagEmbedding.finetune.embedder.encoder_only.base import (
    EncoderOnlyEmbedderDataArguments,
    EncoderOnlyEmbedderTrainingArguments,
    EncoderOnlyEmbedderModelArguments,
    EncoderOnlyEmbedderRunner,
)

parser = HfArgumentParser(
    (
        EncoderOnlyEmbedderModelArguments,
        EncoderOnlyEmbedderDataArguments,
        EncoderOnlyEmbedderTrainingArguments,
    )
)

base_args_d = {
    "model_name_or_path": "BAAI/bge-small-en-v1.5",
    # "train_data": [
    #     "./data/classification-no_in_batch_neg",
    # ],
    "cache_path": "./cache/data",
    "train_group_size": 8,
    "query_max_len": 64,
    "passage_max_len": 64,
    "pad_to_multiple_of": 8,
    "query_instruction_for_retrieval": "Represent this sentence for searching relevant passages: ",
    "query_instruction_format": "{}{}",
    "knowledge_distillation": False,
    # "output_dir": "./test_encoder_only_base_bge-small-en-v1.5",
    "overwrite_output_dir": True,
    "learning_rate": 1e-5,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 3,
    "dataloader_drop_last": True,
    "warmup_ratio": 0.1,
    "logging_steps": 1,
    "save_steps": 500,
    "temperature": 0.02,
    "sentence_pooling_method": "cls",
    "normalize_embeddings": True,
    "kd_loss_type": "kl_div",
}


def get_runner(args_d):
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parser.parse_dict(args_d)
    model_args: EncoderOnlyEmbedderModelArguments
    data_args: EncoderOnlyEmbedderDataArguments
    training_args: EncoderOnlyEmbedderTrainingArguments
    # 在 runner 查看 dataset 对象
    runner = EncoderOnlyEmbedderRunner(
        model_args=model_args, data_args=data_args, training_args=training_args
    )
    return runner
    # runner.run()


if __name__ == "__main__":
    # base_args_d["output_dir"] = "output/no_same_no_batch"

    base_args_d["same_dataset_within_batch"] = True
    base_args_d["train_group_size"] = 8
    base_args_d["train_data"] = [
        "./data/classification",
    ]
    # base_args_d["train_data"] = [
    #     "./data/classification-no_in_batch_neg",
    # ]
    base_args_d["output_dir"] = "output/same"
    ruuner = get_runner(base_args_d)
    ruuner.run()
