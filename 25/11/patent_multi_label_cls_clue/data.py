import json

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# from transformers.data.data_collator import DataCollatorWithPadding
from arguments import PatentDataArgs
from settings import PATENT_CLS_NAMES, PATENT_CLS_ORDER


class PatentDataset(Dataset):
    def __init__(self, json_file):
        self.json_file = json_file
        self._dataset = load_dataset("json", data_files=self.json_file, split="train")
        self.patent_text = self._dataset["patent_text"]
        self.pred_label = self._dataset["pred_label"]

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        patent_text = self.patent_text[idx]
        labels = [0.] * len(PATENT_CLS_NAMES)
        return {"patent_text": patent_text, "labels": labels}

class PatentPredictDataset(Dataset):
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file, usecols=['专利名称','摘要文本', 'nodes'], low_memory=False)
        self.patent_text_template = "专利名: {patent_name}\n专利摘要: {patent_abstract}\n\n候选类别:{clues}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx].to_dict()
        patent_name, patent_abstract, pred_label = item['专利名称'], item["摘要文本"], item["nodes"]
        pred_label = eval(pred_label)
        clue = []
        for cur_name in pred_label:
            label_idx = PATENT_CLS_ORDER.get(cur_name, -1)
            if label_idx != -1:
                clue.append(cur_name)
        prompt = self.patent_text_template.format(
            patent_name=patent_name,
            patent_abstract=patent_abstract,
            clues=" ".join(clue)
        )
        return {"patent_text": prompt}


class PatentCollator:

    def __init__(self, data_args:PatentDataArgs, tokenizer):
        # super().__init__(tokenizer=tokenizer)
        self.data_args = data_args
        self.tokenizer = tokenizer

    def __call__(self, features):
        text = [f["patent_text"] for f in features]
        
        labels = None
        if "labels" in features[0].keys():
            labels = [f["labels"] for f in features]
            labels = torch.tensor(labels)

        text_tokens = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.data_args.text_max_length,
            return_tensors="pt",
        )
        del text_tokens["token_type_ids"]
        
        if labels is not None:
            return {
                **text_tokens,
                "labels": labels,
            }
        return text_tokens

if __name__ == "__main__":
    json_file = "../../data/llm_pred/Qwen2.5-7B-Instruct_pred_2w.jsonl"
    dataset = PatentDataset(json_file)
    print(len(dataset))
    print(dataset[0])

    model_name = "google-bert/bert-base-chinese"
    data_args = PatentDataArgs(patent_pred_json_file=json_file)
    # model_args = PatentModelArgs(model_name_or_path=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = PatentCollator(data_args=data_args, tokenizer=tokenizer)
    collator_data = data_collator([
        dataset[0], dataset[1], dataset[2]
    ])
    print(collator_data)
