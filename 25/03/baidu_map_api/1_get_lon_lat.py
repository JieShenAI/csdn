import os
import sys
import pandas as pd
import asyncio
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import aiohttp
import json
import pandas as pd

from utils import save_json, load_json
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

file = "data/CityList_333.xlsx"
df = pd.read_excel(file)

# 每秒最多 3 个请求
MAX_WORKERS = 2.8  # 根据API限制调整并发数
MAX_ERR_NUM = 5
err_cnt = 0

limiter = AsyncLimiter(MAX_WORKERS, 1)

# 接口配置
API_URL = "https://api.map.baidu.com/geocoding/v3"
AK = os.getenv("api_key")


RETRY_TIMES = 3  # 失败重试次数
TIMEOUT = 10  # 请求超时时间

output_fold = "API_results"
os.makedirs(output_fold, exist_ok=True)
base_name = os.path.basename(file).split(".")[0]
output_file = os.path.join(output_fold, f"{base_name}.json")

ANS = {}
PRE_DATA = {}
if os.path.exists(output_file):
    PRE_DATA = load_json(output_file)


def get_address(row):
    prov = row["省份"].strip()
    city = row["地级行政区"].strip()
    ans = ""
    if prov == city:
        ans = prov
    else:
        ans = prov + city
    return ans


ADDRESSES = df.apply(get_address, axis=1).to_list()


async def geocoding(session, idx, address, progress, retry=RETRY_TIMES):
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
        print(f"Arrive the MAX_ERR_NUM ({str(MAX_ERR_NUM)}), stop running!")
        # 能够被 __main__ 异常捕获住，并完成已有数据保存
        sys.exit(0)

    if address in PRE_DATA.keys():
        progress.update(1)
        ANS.update({address: PRE_DATA[address]})
        return PRE_DATA[address]

    _func_ans = {}

    for _ in range(retry):
        try:
            async with limiter:
                async with session.get(
                    API_URL, params=params, timeout=TIMEOUT
                ) as response:

                    if response.status == 200:
                        data = await response.text()
                        data = json.loads(data)
                        # 添加辅助信息，便于后续数据提取
                        data.update({"idx": idx, "address_input": address})

                        if data.get("status") == 0:
                            progress.update(1)
                            _func_ans = {address: data}
                        else:
                            err_cnt += 1
                            print(data)
                        break

                    elif 500 <= response.status < 600:
                        # 服务器错误,重新发起请求
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError):
            continue
        except Exception as e:
            err_cnt += 1
            print(address, e.args)
            break

    if _func_ans:
        ANS.update(_func_ans)


# # 运行多个 API 调用并保存结果
async def fetch_all():
    tasks = []
    with tqdm(total=len(ADDRESSES), desc="Processing") as progress:
        async with aiohttp.ClientSession() as session:
            for idx, address in enumerate(ADDRESSES):
                if address is None:
                    continue
                tasks.append(geocoding(session, idx, address, progress))
            await asyncio.gather(*tasks)  # 并发执行所有任务


if __name__ == "__main__":
    try:
        asyncio.run(fetch_all())
    except Exception as e:
        print(e.args)
    finally:
        save_json(ANS, output_file)
