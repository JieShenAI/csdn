import asyncio
import os
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import generate_arithmetic_expression, re_parse_json, calculate_time_difference


@dataclass
class LLMAPI:
    """
    cnt: 统计每一个token的调用次数
    """
    api_key: str
    uid: int
    cnt: int = 0
    llm: ChatOpenAI = field(init=False)  # 自动创建的对象，不需要用户传入

    def __post_init__(self):
        # 在这里初始化 llm 对象
        self.llm = self.create_llm()

    def create_llm(self):
        # 模拟创建 llm 对象的逻辑
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url="https://api.chatfire.cn/v1/",
            api_key=self.api_key,
        )

    def invoke(self, text):
        self.cnt += 1
        return self.llm.invoke(text)


def call_llm(llm: LLMAPI, text: str):
    return llm.invoke(text)


def run_api(keys: List[str], data: List[str]) -> Tuple[List[str], List[LLMAPI]]:
    llms = [LLMAPI(api_key=key, uid=i) for i, key in enumerate(keys)]
    results = [call_llm(llms[i % len(llms)], text) for i, text in tqdm(enumerate(data), total=len(data))]
    return results, llms


if __name__ == '__main__':
    start_time = time.time()
    # 加载 .env 文件
    load_dotenv()

    api_keys = os.getenv('API_KEY').split(",")

    prompt_template = """
        请将以下表达式的计算结果返回为 JSON 格式：
        {{
          "expression": "{question}", 
          "result": ? 
        }}
        """

    questions = []
    labels = []
    for _ in range(90):
        question, label = generate_arithmetic_expression(2)
        questions.append(prompt_template.format(question=question))
        labels.append(label)

    results, llms = run_api(api_keys, questions)

    right = 0
    for q, res, label in zip(questions, results, labels):
        res = res.generations[0][0].text
        try:
            res = int(re_parse_json(res).get("result"))
            if res == label:
                right += 1
        except Exception as e:
            print(e)
            print(f"question:{q}\nresult:{res}")

    print("accuracy: {}%".format(right / len(questions) * 100))
    end_time = time.time()
    calculate_time_difference(start_time, end_time)
