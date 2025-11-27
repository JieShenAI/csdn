import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from transformers.trainer_utils import get_last_checkpoint
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer
from arguments import PatentDataArgs, PatentModelArgs
from model import PatentClassifier, compute_metrics
from data import PatentDataset, PatentPredictDataset, PatentCollator
from torch.utils.data import random_split

from settings import PATENT_CLS_NAMES


class PatentTrainer:
    """
    python train.py --config params.json
    python train.py --config predict_params.json
    """
    def __init__(self):
        parser = HfArgumentParser(
            (PatentDataArgs, PatentModelArgs, TrainingArguments)
        )

        # 支持通过 --config xxx.json 传递json格式参数
        if len(sys.argv) == 3 and sys.argv[1] == "--config":
            json_path = sys.argv[2]
            self.data_args, self.model_args, self.training_args = parser.parse_json_file(json_file=json_path)
        else:
            self.data_args, self.model_args, self.training_args = parser.parse_args_into_dataclasses()

        self.best_model_dir = os.path.join(self.training_args.output_dir, "best_model")
        # self.model_args: PatentModelArgs
        self.model = PatentClassifier(self.model_args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)

        if self.data_args.patent_train_json_file:
            self.dataset = PatentDataset(self.data_args.patent_train_json_file)
            self.train_dataset, self.eval_dataset = random_split(
                self.dataset,[self.data_args.train_dataset_size_or_ratio, self.data_args.eval_dataset_size_or_ratio]
            )

    def train(self):
        logging.info("train_dataset length: %d", len(self.train_dataset))
        logging.info(self.train_dataset[0])
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=PatentCollator(data_args=self.data_args, tokenizer=self.tokenizer),
            compute_metrics=compute_metrics
        )
        
        # 检查是否有上次的 checkpoint
        self.last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and not self.training_args.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if self.last_checkpoint is not None:
                logging.info(
                    f"⚡ Found checkpoint at {self.last_checkpoint}. Resuming training."
                )
            else:
                logging.warning("❌ No checkpoint found. Starting fresh training.")
        else:
            logging.info(
                "➡️ No previous output_dir or overwrite_output_dir=True, starting from scratch."
            )
            
        self.trainer.train(
            resume_from_checkpoint=(
                self.last_checkpoint if self.last_checkpoint else None
            )
        )
        self.tokenizer.save_pretrained(self.best_model_dir)
        self.model.model.save_pretrained(self.best_model_dir)

    def predict(self, pred_dir="output"):
        pred_dataset = PatentPredictDataset(csv_file=self.data_args.patent_predict_csv_file)
        logging.info("pred_dataset length: %d", len(pred_dataset))
        logging.info(pred_dataset[0])
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=PatentCollator(data_args=self.data_args, tokenizer=self.tokenizer),
        )
        pred_output = trainer.predict(pred_dataset)
        # 将 logits 转换为概率并得到预测结果
        probs = torch.sigmoid(torch.tensor(pred_output.predictions)).numpy()
        preds = (probs >= 0.5).astype(int)
        preds_df = pd.DataFrame(preds, columns=PATENT_CLS_NAMES)
        pred_file = os.path.join(pred_dir, os.path.basename(self.data_args.patent_predict_csv_file).replace(".csv", "_pred.csv"))
        preds_df = pd.concat([pred_dataset.df, preds_df], axis=1)
        preds_df.to_csv(pred_file, index=False, encoding="utf-8-sig")
        logging.info(f"Predictions saved to {pred_file}")
        

if __name__ == "__main__":
    patent_trainer = PatentTrainer()
    # patent_trainer.train()
    patent_trainer.predict()