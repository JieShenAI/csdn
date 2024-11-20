import asyncio
import os
import time

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple, TypedDict
from aiolimiter import AsyncLimiter

# 创建限速器，每秒最多发出 5 个请求
limiter = AsyncLimiter(5, 1)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import (
    generate_arithmetic_expression,
    re_parse_json,
    calculate_time_difference,
)


@dataclass
class LLMAPI:
    """
    大模型API的调用类
    """
    base_url: str
    api_key: str  # 每个API的key不一样
    uid: int
    cnt: int = 0  # 统计每个API被调用了多少次
    llm: ChatOpenAI = field(init=False)  # 自动创建的对象，不需要用户传入

    def __post_init__(self):
        # 初始化 llm 对象
        self.llm = self.create_llm()

    def create_llm(self):
        # 创建 llm 对象
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def agenerate(self, text):
        self.cnt += 1
        res = await self.llm.agenerate([text])
        return res


async def call_llm(llm: LLMAPI, text: str):
    # 异步协程 限速
    async with limiter:
        res = await llm.agenerate(text)
        return res


async def _run_task_with_progress(task, pbar):
    """包装任务以更新进度条"""
    result = await task
    pbar.update(1)
    return result


async def run_api(llms: List[LLMAPI], data: List[str]) -> Tuple[List[str], List[LLMAPI]]:
    results = [call_llm(llms[i % len(llms)], text) for i, text in enumerate(data)]

    # 使用 tqdm 创建一个进度条
    with tqdm(total=len(results)) as pbar:
        # 使用 asyncio.gather 并行执行任务
        results = await asyncio.gather(
            *[_run_task_with_progress(task, pbar) for task in results]
        )
    return results, llms


if __name__ == "__main__":
    load_dotenv()

    # 四则运算提示词模板
    prompt_template = """
    请将以下表达式的计算结果返回为 JSON 格式：
    {{
      "expression": "{question}",
      "result": ?
    }}
    """

    questions = []
    labels = []

    for _ in range(100):
        question, label = generate_arithmetic_expression(2)
        questions.append(prompt_template.format(question=question))
        labels.append(label)

    start_time = time.time()

    # for jupyter
    # results, llms = await run_api(api_keys, questions)

    api_keys = os.getenv("API_KEY").split(",")
    base_url = os.getenv("BASE_URL")
    # 创建LLM
    llms = [LLMAPI(base_url=base_url, api_key=key, uid=i) for i, key in enumerate(api_keys)]
    results, llms = asyncio.run(run_api(llms, questions))

    right = 0  # 大模型回答正确
    except_cnt = 0  # 大模型不按照json格式返回结果
    not_equal = 0  # 大模型解答错误

    for q, res, label in zip(questions, results, labels):
        res = res.generations[0][0].text
        try:
            res = re_parse_json(res)
            if res is None:
                except_cnt += 1
                continue

            res = res.get("result", None)
            if res is None:
                except_cnt += 1
                continue

            res = int(res)
            if res == label:
                right += 1
            else:
                not_equal += 1
        except Exception as e:
            print(e)
            print(f"question:{q}\nresult:{res}")

    print("accuracy: {}%".format(right / len(questions) * 100))
    end_time = time.time()
    calculate_time_difference(start_time, end_time)
    print(right, except_cnt, not_equal)

    print(
        f"Question: {questions[0]}\n"
        f"LLM Infer: {results[0].generations[0][0].text}\n"
        f"label: {labels[0]}"
    )
