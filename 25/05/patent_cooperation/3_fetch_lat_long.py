import os
import sys
import pandas as pd
import asyncio
from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm_asyncio
import aiohttp
import json
import pandas as pd

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

import json
from typing import Dict


def save_json(data: Dict, file: str):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file) -> Dict:
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


file = "output/2505/all_companies.csv"
address_col_name = "companies"
df = pd.read_csv(file)


# 每秒最多 3 个请求
MAX_WORKERS = 2.8  # 根据API限制调整并发数
MAX_ERR_NUM = 20
err_cnt = 0

limiter = AsyncLimiter(MAX_WORKERS, 1)

# 接口配置
API_URL = "https://api.map.baidu.com/geocoding/v3"
AK = os.getenv("api_key")


TIMEOUT = 10  # 请求超时时间

output_fold = "API_results"
os.makedirs(output_fold, exist_ok=True)
base_name = os.path.basename(file).split(".")[0]
output_file = os.path.join(output_fold, f"{base_name}.json")

ANS = {}
PRE_DATA = {}
if os.path.exists(output_file):
    PRE_DATA = load_json(output_file)


ADDRESSES = df[address_col_name].to_list()


class MaxErrorExceededException(Exception):
    """自定义异常类，用于表示错误次数超过最大限制"""
    pass


async def geocoding(session, idx, address):
    """地理编码请求函数（带重试机制）"""
    params = {
        "address": address,
        "output": "json",
        "ak": AK,
        # "ret_coordtype": "gcj02ll",  # 可根据需要修改坐标系, 国测局坐标
    }
    global err_cnt
    global ANS
    global PRE_DATA

    if err_cnt >= MAX_ERR_NUM:
        raise MaxErrorExceededException(
            f"Error count exceeded the maximum limit: {MAX_ERR_NUM}"
        )
        # sys.exit(0)

    if address in PRE_DATA.keys():
        # progress.update(1)
        ANS.update({address: PRE_DATA[address]})
        # return PRE_DATA[address]
        return

    _func_ans = {}

    try:
        async with limiter:
            async with session.get(API_URL, params=params, timeout=TIMEOUT) as response:
                if response.status == 200:
                    data = await response.text()
                    data = json.loads(data)
                    # 添加辅助信息，便于后续数据提取
                    data.update({"idx": idx, "address_input": address})

                    if data.get("status") == 0:
                        # progress.update(1)
                        _func_ans = {address: data}
                    else:
                        err_cnt += 1
                        print(data)
                else:
                    err_cnt += 1
    except Exception as e:
        err_cnt += 1
        print(address, e.args)

    if _func_ans:
        ANS.update(_func_ans)


# # 运行多个 API 调用并保存结果
async def fetch_all():
    tasks = []
    async with aiohttp.ClientSession() as session:
        for idx, address in enumerate(ADDRESSES):
            if address is None:
                continue
            tasks.append(geocoding(session, idx, address))
        await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":

    try:
        asyncio.run(fetch_all())
    except Exception as e:
        print(e.args)
    finally:
        save_json(ANS, output_file)
