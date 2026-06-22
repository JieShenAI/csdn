import torch
from transformers.data.data_collator import DataCollatorWithPadding

from .arguments import DataArguments


class DiyDataCollator(DataCollatorWithPadding):

    def __init__(self, data_args: DataArguments, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.data_args = data_args

    def __call__(self, features):
        texts = [f["text"] for f in features]
        labels = [f["label"] for f in features]
        labels = torch.tensor(labels)

        text_tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.data_args.text_max_length,
            return_tensors="pt",
        )

        return {
            "text_tokens": text_tokens,
            "labels": labels,
        }
