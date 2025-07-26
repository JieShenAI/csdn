"""
目的：想把一个纯英文的模型，让它学会汉语，再学会从英文翻译到汉语。（没有实现，最后放弃了）
挑战: 发现在模型微调的过程中，会出现梯度消失、梯度爆炸、loss为0等现象。
想法： 发现直接微调模型学会从英文翻译中文的过程，我认为是初始的embedding的参数不够好。】
于是便想让模型在中文数据集做预训练从而调整新增加的中文token的向量表示，但是发现在英文t5模型在中文数据集预训练的过程中，
也会出现梯度消失、梯度爆炸、loss为0等现象。于是我只能放弃了！
TODO: 希望后面有能力的时候，再回来继续尝试！
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np

# 设置镜像地址
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

# SDK模型下载
from modelscope import snapshot_download

model_dir = snapshot_download("AI-ModelScope/t5-base")

# model_name = "/home/valiantsec/phb/models/flan-t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)


from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import evaluate

from tqdm import tqdm
from datasets import load_dataset

dataset = load_dataset("Helsinki-NLP/news_commentary", "en-zh", trust_remote_code=True)


import re


def is_chinese_char(char):
    return bool(re.match(r"[\u4e00-\u9fff]", char))


def get_only_word(raw_dataset):
    """
    拿到每一个唯一的中文字
    """
    words = set()
    for item in raw_dataset:
        text = item["translation"]["zh"]
        for word in text:
            if is_chinese_char(word):
                words.add(word)
    return words


zh_words = get_only_word(dataset["train"])


for zh_word in tqdm(zh_words):
    tokenizer.add_tokens(zh_word)


model.resize_token_embeddings(len(tokenizer))


def process_dataset(batch_item):
    # item = item["translation"]
    # en_text = item["en"]
    # zh_text = item["zh"]
    en_texts = [t["en"] for t in batch_item["translation"]]
    zh_texts = [t["zh"] for t in batch_item["translation"]]
    en_tokens = tokenizer(
        en_texts,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    zh_tokens = tokenizer(
        zh_texts,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    en_tokens["labels"] = zh_tokens["input_ids"]
    en_tokens["labels"][en_tokens["labels"] == tokenizer.pad_token_id] = -100
    return en_tokens


train_dataset, eval_dataset = dataset["train"].train_test_split(test_size=0.2).values()


train_dataset = train_dataset.map(
    process_dataset,
    batched=True,
    num_proc=1,
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

eval_dataset = eval_dataset.map(
    process_dataset,
    batched=True,
    num_proc=1,
    remove_columns=eval_dataset.column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)


# data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
seq2seq_test_data = data_collator(
    (train_dataset[0], train_dataset[1], train_dataset[2])
)

# 加载ROUGE评估指标
# metric = load_metric("rouge")
metric = evaluate.load("rouge")

result = metric.compute(
    predictions=["hello world"],
    references=["hello new world"],
    use_stemmer=True,  # 使用词干提取以提高匹配度
)


def compute_metrics(eval_pred):
    """生成式任务的评估函数"""
    predictions, labels = eval_pred
    predictions = predictions[0].argmax(-1)

    # 解码生成的文本（模型预测）
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 解码目标文本（真实标签）
    # 替换-100为pad_token_id以正确解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 可选：简单后处理（如去除换行符）
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    right = 0
    for pred, label in zip(decoded_preds, decoded_labels):
        if pred == label:
            right += 1

    # 计算ROUGE分数
    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,  # 使用词干提取以提高匹配度
    )

    # 提取主要ROUGE指标（如ROUGE-1、ROUGE-2、ROUGE-L）
    # 取平均值作为最终结果
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


training_args = TrainingArguments(
    "output/en_zh",
    evaluation_strategy="epoch",
    # learning_rate=3e-4,
    gradient_accumulation_steps=1,
    auto_find_batch_size=True,  # 自动设置batch_size（学习）
    # per_device_train_batch_size=64,
    # per_device_eval_batch_size=32,
    num_train_epochs=1,
    # save_strategy="epoch",
    save_strategy="steps",  # 改为每 N 步保存
    save_steps=1000,  # 每 500 步保存一次
    save_total_limit=3,
    logging_strategy="steps",  # 按步骤记录日志
    logging_steps=100,  # 每10步记录一次训练损失
    label_names=["labels"],
    learning_rate=5e-5,  # 更安全
    warmup_steps=200,  # 推荐加上
    weight_decay=0.01,  # 可提高泛化能力
    fp16=True,
)


model.train()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)


trainer.train()
