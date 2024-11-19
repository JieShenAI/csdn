import random
import asyncio
from uuid import uuid4
from tqdm import tqdm
from dataclasses import dataclass
from aiolimiter import AsyncLimiter

# 创建限速器，每秒最多发出 5 个请求
limiter = AsyncLimiter(10, 1)


@dataclass
class Token:
    uid: str
    idx: int
    cnt: int = 0


# 将 connect_web 改为异步函数
async def llm_api(data):
    t = random.randint(0, 2)
    await asyncio.sleep(t)  # 使用 asyncio.sleep 替代 time.sleep
    # time.sleep(8)
    await asyncio.sleep(8)  # 使用 asyncio.sleep 替代 time.sleep
    return data * 10


# 保持 call_api 异步
async def call_api(token, data, rate_limit_seconds=0.5):
    token.cnt += 1
    async with limiter:
        await asyncio.sleep(rate_limit_seconds)
        return await llm_api(data)


workders = 1
tokens = [Token(uid=str(uuid4()), idx=i) for i in range(workders)]


async def _run_task_with_progress(task, pbar):
    """包装任务以更新进度条"""
    result = await task
    pbar.update(1)
    return result


# 主函数
async def main():
    nums = 100
    data = [i for i in range(nums)]
    results = [call_api(tokens[int(i % workders)], item) for i, item in enumerate(data)]
    # 使用 asyncio.gather 调用异步任务
    # results = await asyncio.gather(*result)

    # 使用 tqdm 创建一个进度条
    with tqdm(total=len(results)) as pbar:
        # 使用 asyncio.gather 并行执行任务
        results = await asyncio.gather(
            *(_run_task_with_progress(task, pbar) for task in results)
        )
    return results


# 运行程序
result = asyncio.run(main())
print(result)
