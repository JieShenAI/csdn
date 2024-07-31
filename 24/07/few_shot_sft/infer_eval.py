import os
import json
import random
import logging
import argparse
import pickle
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.FileHandler('../eval.log')],
    level=logging.INFO
)


@dataclass
class EvalData:
    name : str
    in_cnt : int = 0
    not_in_cnt : int = 0
    preds : list = field(default_factory=list)
    labels : list = field(default_factory=list)
    not_in_texts : list = field(default_factory=list)
    eval : dict = field(default_factory=dict)

def save_obj(obj, name):  
    """  
    将对象保存到文件  
    :param obj: 要保存的对象  
    :param name: 文件的名称（包括路径）  
    """  
    with open(name, 'wb') as f:  
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):  
    """  
    从文件加载对象  
    :param name: 文件的名称（包括路径）  
    :return: 反序列化后的对象  
    """  
    with open(name, 'rb') as f:  
        return pickle.load(f)


LABELS_DICT = {
    0: "Human Necessities",
    1: "Performing Operations; Transporting",
    2: "Chemistry; Metallurgy",
    3: "Textiles; Paper",
    4: "Fixed Constructions",
    5: "Mechanical Engineering; Lightning; Heating; Weapons; Blasting",
    6: "Physics",
    7: "Electricity",
    8: "General tagging of new or cross-sectional technology",
}


LABELS_NAME = [
    LABELS_DICT[i]
    for i in range(9)
]

LABELS_2_IDS = {
    v : k
    for k, v in LABELS_DICT.items()
}


def compute_metrics(pred, label):
    res = {}
    accuracy = evaluate.load("accuracy")
    res.update(accuracy.compute(
            predictions=pred, 
            references=label
        ))

    precision = evaluate.load("precision")
    res.update(precision.compute(
            predictions=pred, 
            references=label,
            average="macro"
        ))

    recall = evaluate.load("recall")
    res.update(recall.compute(
            predictions=pred, 
            references=label,
            average="macro"
        ))

    f1 = evaluate.load("f1")
    res.update(f1.compute(
            predictions=pred, 
            references=label,
            average="macro"
        ))
    return res


def eval(kw):
    eval_data = EvalData(name=kw)
    model = ChatOpenAI(
        api_key="0",
        base_url="http://localhost:8000/v1",
        temperature=0
    )

    valid_dataset = load_dataset(
        "json",
        data_files="../data/llm_valid.json"
    )["train"]
    # labels = valid_dataset["output"][:50]
    labels = valid_dataset["output"]
    
    eval_data.labels = labels
    
    parser = StrOutputParser()
    preds = []
    cnt = 0
    for item in tqdm(valid_dataset):
        cnt += 1
        messages = [
            SystemMessage(content=item['instruction']),
            HumanMessage(content=item['input']),
        ]
        chain = model | parser
        pred = chain.invoke(messages).strip()
        preds.append(pred)
        # if cnt == 50:
        #     break
    
    eval_data.preds = preds

    not_in_texts = []
    in_cnt = 0
    not_in_cnt = 0

    for pred in preds:
        if pred in LABELS_NAME:
            in_cnt += 1
        else:
            not_in_cnt += 1
            not_in_texts.append(pred)
    
    eval_data.in_cnt = in_cnt
    eval_data.not_in_cnt = not_in_cnt
    eval_data.not_in_texts = not_in_texts
    
    pred_num = [
        LABELS_2_IDS[pred] if pred in LABELS_NAME else random.choice(range(9))
        for pred in preds
    ]
    label_num = [
        LABELS_2_IDS[label]
        for label in labels
    ]
    
    eval_data.eval = compute_metrics(pred=pred_num, label=label_num)
    
    logging.info(f"in_cnt: {in_cnt}, not_in_cnt: {not_in_cnt}")
    logging.info(f"eval: {eval_data.eval}")
    
    # 推理结果保存
    save_obj(
            eval_data,
            f"../objs/{kw}.pkl"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="输入大模型名，开始推理")
    parser.add_argument("kw", help="目前部署的大模型名字")
    args = parser.parse_args()
    logging.info(args.kw)
    eval(args.kw)