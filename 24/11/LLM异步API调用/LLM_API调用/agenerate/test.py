import asyncio
import os
import time

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Tuple
from aiolimiter import AsyncLimiter

# 创建限速器，每秒最多发出 5 个请求
limiter = AsyncLimiter(6, 1)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from utils import (
    generate_arithmetic_expression,
    re_parse_json,
    calculate_time_difference,
    AsyncLLMAPI,
)


load_dotenv()


prompt_template = """
你是一名擅长数学运算的助手，负责逐步推理并解决四则运算问题。请按照以下步骤进行：

1. 阅读并理解问题。
2. 分步计算，逐步解决问题。
3. 给出最终的结果。
4. 按照 JSON 格式输出结果，包括：
- reason: 详细的推理过程。
- infer: 最终的计算结果。

问题：{question}
请给出分析和结果。
""".strip()

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

llms = [
    AsyncLLMAPI(base_url=base_url, api_key=key, uid=i) for i, key in enumerate(api_keys)
]

# results = [llms[i % len(api_keys)](text) for i, text in enumerate(api_keys)]

results, llms = AsyncLLMAPI.run_data_async(llms=llms, data=questions)
