import re
import json
import random
import time
from typing import Union, Dict
import asyncio
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List
from aiolimiter import AsyncLimiter
from langchain_openai import ChatOpenAI


def generate_arithmetic_expression(num: int):
    """
    num: 几个操作符
    """
    # 定义操作符和数字范围，除法
    operators = ["+", "-", "*"]
    expression = (
        f"{random.randint(1, 100)} {random.choice(operators)} {random.randint(1, 100)}"
    )
    num -= 1
    for _ in range(num):
        expression = f"{expression} {random.choice(operators)} {random.randint(1, 100)}"
    result = eval(expression)
    expression = expression.replace("*", "x")
    return expression, result


# 假设这是模型返回的内容
model_response = """
Here is your calculation:
{
  "expression": "70 + 81 - 43 + 48 + 9",
  "result": 165
}
"""


def re_parse_json(text) -> Union[Dict, None]:
    # 提取 JSON 内容
    json_match = re.search(r"\{.*?\}", text, re.DOTALL)
    if json_match:
        json_data = json_match.group(0)
        response_data = json.loads(json_data)
        return response_data
    print(f"异常:\n{text}")
    return None


def calculate_time_difference(start_time, end_time):
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    milliseconds = (elapsed_time - int(elapsed_time)) * 1000

    print(
        f"executed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03} (h:m:s.ms)"
    )


def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间

        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        milliseconds = (elapsed_time - int(elapsed_time)) * 1000

        print(
            f"Function '{func.__name__}' executed in {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03} (h:m:s.ms)"
        )
        return result

    return wrapper


"""
pip install langchain tqdm aiolimiter python-dotenv
"""


@dataclass
class AsyncLLMAPI:
    """
    大模型API的调用类
    """

    base_url: str
    api_key: str  # 每个API的key不一样
    uid: int
    cnt: int = 0  # 统计每个API被调用了多少次
    llm: ChatOpenAI = field(init=False)  # 自动创建的对象，不需要用户传入
    num_per_second: int = 6

    def __post_init__(self):
        # 初始化 llm 对象
        self.llm = self.create_llm()
        # 创建限速器，每秒最多发出 5 个请求
        self.limiter = AsyncLimiter(self.num_per_second, 1)

    def create_llm(self):
        # 创建 llm 对象
        return ChatOpenAI(
            model="gpt-4o-mini",
            base_url=self.base_url,
            api_key=self.api_key,
        )

    async def __call__(self, text):
        # 异步协程 限速
        self.cnt += 1
        async with self.limiter:
            return await self.llm.agenerate([text])

    @staticmethod
    async def _run_task_with_progress(task, pbar):
        """包装任务以更新进度条"""
        result = await task
        pbar.update(1)
        return result
    
    @staticmethod
    def run_data_async(llms: List["AsyncLLMAPI"], data: List[str]):

        async def _sync_run(llms, data):
            results = [llms[i % len(llms)](text) for i, text in enumerate(data)]
            # 使用 tqdm 创建一个进度条
            with tqdm(total=len(results)) as pbar:
                # asyncio.gather 并行执行任务
                results = await asyncio.gather(
                    *[
                        AsyncLLMAPI._run_task_with_progress(task, pbar)
                        for task in results
                    ]
                )
            return results, llms

        return asyncio.run(_sync_run(llms, data))


# 测试生成
if __name__ == "__main__":
    expr, res = generate_arithmetic_expression(4)
    print(f"生成的运算表达式: {expr}")
    print(f"计算结果: {res}")
