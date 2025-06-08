import sys

sys.path.append("..")

import logging
import os
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from src.arguments import (
    ModelArguments,
    DataArguments,
    RetrieverTrainingArguments as TrainingArguments,
)
from src.data import TrainDatasetForEmbedding, EmbedCollator
from src.modeling import BiEncoderModel

# from trainer import BiTrainer
from transformers import Trainer

# Model Download
from modelscope import snapshot_download

model_dir = snapshot_download("AI-ModelScope/bert-base-uncased")

args_d = {
    "output_dir": "output",
    # "model_name_or_path": "BAAI/bge-large-zh-v1.5",
    "model_name_or_path": model_dir,
    "train_data": "./toy_finetune_data.jsonl",
    "learning_rate": 1e-5,
    "fp16": True,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 2,
    "dataloader_drop_last": True,
    "normlized": True,
    "temperature": 0.02,
    "query_max_len": 64,
    "passage_max_len": 256,
    "train_group_size": 4,
    "negatives_cross_device": False,
    "logging_steps": 10,
    "query_instruction_for_retrieval": "Generate representations for this sentence to retrieve related articles:",
    "save_safetensors": False,
}

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_dict(args_d)
