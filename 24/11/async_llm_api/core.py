import os
import random
import asyncio
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass, field
from aiolimiter import AsyncLimiter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def generate_arithmetic_expression(num: int):
    """
    生成数学计算的

    num: 几个操作符

        prompt_template:
        你是一名擅长数学运算的助手，负责逐步推理并解决四则运算问题。请按照以下步骤进行：

        1. 阅读并理解问题。
        2. 分步计算，逐步解决问题。
        3. 给出最终的结果。
        4. 按照 JSON 格式输出结果，包括：
        - reason: 详细的推理过程。
        - infer: 最终的计算结果。

        问题：{question}
        请给出分析和结果。
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


@dataclass
class AsyncLLMAPI:
    """
    大模型API的调用类
    """

    base_url: str
    api_key: str  # 每个API的key不一样
    uid: int = 0
    cnt: int = 0  # 统计每个API被调用了多少次
    model: str = "gpt-3.5-turbo"
    llm: ChatOpenAI = field(init=False)  # 自动创建的对象，不需要用户传入
    num_per_second: int = 6  # 限速每秒调用6次

    def __post_init__(self):
        # 初始化 llm 对象
        self.llm = self.create_llm()
        # 创建限速器，每秒最多发出 5 个请求
        self.limiter = AsyncLimiter(self.num_per_second, 1)

    def create_llm(self):
        # 创建 llm 对象
        return ChatOpenAI(
            model=self.model,
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
    def async_run(
        llms: List["AsyncLLMAPI"],
        data: List[str],
        keyword: str = "",  # 文件导出名
        output_dir: str = "output",
        chunk_size=500,
    ):

        async def _func(llms, data):
            """
            异步请求处理一小块数据
            """
            results = [llms[i % len(llms)](text) for i, text in enumerate(data)]
            with tqdm(total=len(results)) as pbar:
                results = await asyncio.gather(
                    *[
                        AsyncLLMAPI._run_task_with_progress(task, pbar)
                        for task in results
                    ]
                )
            return results

        idx = 0
        all_df = []
        while idx < len(data):
            file = f"{idx}_{keyword}.csv"
            file_dir = os.path.join(output_dir, file)

            if os.path.exists(file_dir):
                print(f"{file_dir} already exist! Just skip.")
                tmp_df = pd.read_csv(file_dir)
            else:
                tmp_data = data[idx : idx + chunk_size]

                loop = asyncio.get_event_loop()
                tmp_result = loop.run_until_complete(_func(llms=llms, data=tmp_data))
                tmp_result = [item.generations[0][0].text for item in tmp_result]
                tmp_df = pd.DataFrame({"infer": tmp_result})

                # 如果文件夹不存在，则创建
                if not os.path.exists(tmp_folder := os.path.dirname(file_dir)):
                    os.makedirs(tmp_folder)

                tmp_df.to_csv(file_dir, index=False)

            all_df.append(tmp_df)
            idx += chunk_size

        all_df = pd.concat(all_df)
        all_df.to_csv(os.path.join(output_dir, f"all_{keyword}.csv"), index=False)
        return all_df


if __name__ == "__main__":

    # 生成 数学计算数据集

    texts = []
    labels = []

    for _ in range(1000):
        text, label = generate_arithmetic_expression(2)
        texts.append(text)
        labels.append(label)

    llm = AsyncLLMAPI(
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        num_per_second=10,
    )

    AsyncLLMAPI.async_run(
        [llm], texts, keyword="数学计算", output_dir="output", chunk_size=500
    )
