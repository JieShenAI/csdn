from dataclasses import dataclass
import torch
from transformers import DataCollatorWithPadding, AutoTokenizer

from hf import model_args, data_args, training_args

import sys

sys.path.append("../")

from src.data import TrainDatasetForEmbedding, EmbedCollator

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

data_collator = EmbedCollator(
    tokenizer=tokenizer,
    query_max_len=data_args.query_max_len,
    passage_max_len=data_args.passage_max_len,
)

if __name__ == "__main__":
    head_data = [dataset[0], dataset[1], dataset[2]]
    query, passage = data_collator(head_data).values()
    print(passage["input_ids"].shape)
