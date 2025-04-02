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

file = "data/路径规划_数据.csv"
df = pd.read_csv(file, low_memory=False)


MAX_ERR_NUM = 200  # 最大的错误次数
err_cnt = 0

MAX_WORKERS = 2  # 根据API限制调整并发数, 每秒最多 3 个请求
limiter = AsyncLimiter(MAX_WORKERS, 1)

# 接口配置
BASE_URL = "https://api.map.baidu.com/directionlite/v1/driving"

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


def get_params(
    origin_lat,
    origin_lon,
    target_lat,
    target_lon,
):
    lon_lat_template = "{:.6f},{:.6f}"
    params = {
        "origin": lon_lat_template.format(origin_lat, origin_lon),
        "destination": lon_lat_template.format(target_lat, target_lon),
        "ak": AK,
        "steps_info": 0,
    }
    return params


async def geocoding(session, params, address_key, idx, progress, retry=RETRY_TIMES):
    """地理编码请求函数（带重试机制）"""

    global err_cnt
    global ANS
    global PRE_DATA

    if address_key in PRE_DATA.keys():
        progress.update(1)
        ANS.update({address_key: PRE_DATA[address_key]})
        return

    _func_ans = {}

    for _ in range(retry):
        try:
            async with limiter:
                async with session.get(
                    BASE_URL, params=params, timeout=TIMEOUT
                ) as response:

                    if response.status == 200:
                        data = await response.text()
                        data = json.loads(data)
                        # 添加辅助信息，便于后续数据提取
                        data.update({"idx": idx, "address_input": address_key})

                        if data.get("status") == 0:
                            progress.update(1)
                            _func_ans = {address_key: data}
                        elif data.get("status") == 302:
                            print(f"err_cnt:{err_cnt}", data)
                            sys.exit(0)
                        else:
                            err_cnt += 1
                            print(f"err_cnt:{err_cnt}", data)
                        break

                    elif 500 <= response.status < 600:
                        # 服务器错误,重新发起请求
                        continue
        except (aiohttp.ClientError, asyncio.TimeoutError):
            continue
        except Exception as e:
            err_cnt += 1
            print(address_key, e.args)
            break

    if _func_ans:
        ANS.update(_func_ans)

    if err_cnt >= MAX_ERR_NUM:
        print(f"Arrive the MAX_ERR_NUM ({str(MAX_ERR_NUM)}), stop running!")
        # 能够被 __main__ 异常捕获住，并完成已有数据保存
        sys.exit(0)


# # 运行多个 API 调用并保存结果
async def fetch_all():
    tasks = []
    with tqdm(total=df.shape[0], desc="Processing") as progress:
        async with aiohttp.ClientSession() as session:
            # 海南省
            for idx, row in df.iterrows():
                # if row["origin_prov"] == "海南省" or row["target_prov"] == "海南省":
                #     continue
                origin_lon = row["origin_经度"]
                origin_lat = row["origin_纬度"]
                target_lon = row["target_经度"]
                target_lat = row["target_纬度"]
                params = get_params(
                    origin_lat=origin_lat,
                    origin_lon=origin_lon,
                    target_lat=target_lat,
                    target_lon=target_lon,
                )
                address_key = (
                    (row["origin_prov"], row["origin_city"]),
                    (
                        row["target_prov"],
                        row["target_city"],
                    ),
                )

                tasks.append(
                    geocoding(
                        session=session,
                        params=params,
                        address_key=str(address_key),
                        idx=idx,
                        progress=progress,
                    )
                )
            # 并发执行所有任务
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(fetch_all())
    except Exception as e:
        print(e.args)
    finally:
        save_json(ANS, output_file)
