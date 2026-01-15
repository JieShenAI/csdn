import logging
import os
import sys
from typing import Union, Literal
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import HfArgumentParser, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .arguments import DataArguments, ModelArguments
from .model import DiyConfig, DiyModel


def compute_metrics(eval_pred):
    """
    计算文本分类任务的评估指标

    Args:
        eval_pred: 包含预测结果和真实标签的EvalPrediction对象
            - predictions: 模型输出的原始logits
            - label_ids: 真实标签
        eval_pred: 包含预测结果和真实标签的EvalPrediction对象
            - predictions: 模型输出的原始logits
            - label_ids: 真实标签

    Returns:
        包含评估指标的字典
    """
    # 从EvalPrediction对象中获取预测结果和真实标签
    predictions, label = eval_pred
    # 将logits转换为预测类别（取概率最大的类别）
    preds = np.argmax(predictions, axis=-1)

    # 计算准确率
    accuracy = accuracy_score(label, preds)

    # 计算精确率、召回率和F1分数（支持多类别和二分类）
    # average参数：'micro'、'macro'、'weighted'或None
    precision, recall, f1, _ = precision_recall_fscore_support(
        label, preds, average="weighted"
    )

    # 返回评估指标字典
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


class TrainerUtil:
    def __init__(self, model_class=None, dataset_class=None, datacollator_class=None, trainer_class=Trainer):
        self.last_checkpoint = None
        self.dataset_class = dataset_class
        self.model_class = model_class
        self.datacollator_class = datacollator_class
        self.trainer_class = trainer_class
        self.start_up()
        self.set_dataset()
        self.set_model()

        self.logger.info(self.train_dataset[0])
        self.logger.info(
            f"train on {len(self.train_dataset)} samples, eval on {len(self.eval_dataset)} samples"
        )
        self.trainer = self.trainer_class(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.datacollator_class(
                data_args=self.data_args, tokenizer=self.tokenizer
            ),
            compute_metrics=compute_metrics,
        )

    def set_model(self):
        raise NotImplementedError

    def set_dataset(self):
        """
        若数据集加载的方式不一样，需要重写该函数
        :return:
        """
        self.train_dataset = self.dataset_class(args=self.data_args, dataset_file="train.parquet")
        self.eval_dataset = self.dataset_class(args=self.data_args, dataset_file="validate.parquet")
        self.test_dataset = self.dataset_class(args=self.data_args, dataset_file="test.parquet")

    def start_up(self):
        self.logger = logging.getLogger()
        self.general_formatter = logging.Formatter("%(asctime)s - %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.general_formatter)
        self.logger.addHandler(stream_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("root logger")

        parser = HfArgumentParser(
            (DataArguments, ModelArguments, TrainingArguments)
        )

        if len(sys.argv) == 3 and sys.argv[1] == "--config":
            json_path = sys.argv[2]
            self.data_args, self.model_args, self.training_args = parser.parse_json_file(json_file=json_path)
        else:
            self.data_args, self.model_args, self.training_args = parser.parse_args_into_dataclasses()

        self.data_args: DataArguments
        self.model_args: ModelArguments
        self.training_args: TrainingArguments
        self.best_model_dir = os.path.join(self.training_args.output_dir, "best_model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path
        )

        # if "qwen" in self.model_args.model_name_or_path.lower():
        #     self.model.model.config.pad_token_id = self.tokenizer.pad_token_id

    def save(self, save_model_dir):
        self.tokenizer.save_pretrained(save_model_dir)
        self.model.save_pretrained(save_model_dir)

    def train(self):
        # 检查是否有上次的 checkpoint
        self.last_checkpoint = None
        if (
                os.path.isdir(self.training_args.output_dir)
                and not self.training_args.overwrite_output_dir
        ):
            self.last_checkpoint = get_last_checkpoint(self.training_args.output_dir)

            if self.last_checkpoint is not None:
                self.logger.info(
                    f"⚡ Found checkpoint at {self.last_checkpoint}. Resuming training."
                )
            else:
                self.logger.warning("❌ No checkpoint found. Starting fresh training.")

        else:
            self.logger.info(
                "➡️ No previous output_dir or overwrite_output_dir=True, starting from scratch."
            )

        self.trainer.train(
            resume_from_checkpoint=(
                self.last_checkpoint if self.last_checkpoint else None
            )
        )
        self.save(self.best_model_dir)

    def evaluate(self, dataset: Union[Literal["eval", "test"], datasets.Dataset]):
        eval_logger = logging.getLogger("evaluate")
        file_handler = logging.FileHandler(
            os.path.join(self.training_args.output_dir, "eval.log"),
            encoding="utf-8",
            mode="a+"
        )
        file_handler.setFormatter(self.general_formatter)
        eval_logger.addHandler(file_handler)
        eval_logger.setLevel(logging.INFO)

        if isinstance(dataset, str):
            if dataset not in ("eval", "test"):
                raise ValueError(
                    f"字符串类型的dataset只能是'eval'或'test'，但收到：{dataset}"
                )
            if dataset == "eval":
                metric = self.trainer.evaluate(self.eval_dataset)
                eval_logger.info(
                    f"<<<<< Eval on eval_dataset *****"
                )
            elif dataset == "test":
                metric = self.trainer.evaluate(self.test_dataset)
                eval_logger.info(
                    f"<<<<< Eval on test_dataset *****"
                )
        else:
            metric = self.trainer.evaluate(dataset)

        eval_logger.info(f"Model: {self.model_args.model_name_or_path}")
        eval_logger.info(f"Metrics: {metric}")
        eval_logger.info(f"***** End Eval >>>>>\n")
        return metric


class DiyTrainerUtil(TrainerUtil):
    def set_dataset(self):
        assert self.dataset_class is None, "loading dataset from huggingface, no need for dataset_class."
        dataset = load_dataset(self.data_args.dataset_name)
        train_eval_dataset = dataset["train"].train_test_split(train_size=0.8, seed=42)
        self.train_dataset = train_eval_dataset["train"]
        self.eval_dataset = train_eval_dataset["test"]
        self.test_dataset = dataset["test"]
        self.data_args.num_label = len(set(self.train_dataset["label"]))

    def set_model(self):
        if self.training_args.do_train:
            assert self.model_class is None, "loading custom model, no need for model_class."
            diy_config = DiyConfig(
                hf_name=self.model_args.model_name_or_path,
                num_label=self.data_args.num_label
            )
            self.model = DiyModel(diy_config)
        else:
            # 评估的时候，需要加载本地模型
            self.model = DiyModel.from_pretrained(self.model_args.model_name_or_path)

        print(self.model)
