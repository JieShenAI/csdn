"""
协程调用高德地图API，获取对应的经纬度
1. 从 api_result_dir/pre 文件夹加载之前已经保存的文件
2. api_result_dir/cur/ 保存当前数据

在程序运行完成之后，一定要把 cur/ 里面的文件给转移到 pre文件夹下
ADDRESSES 筛选出在 api_result_dir/pre 还不包含的地址

只要进行了请求，会保存错误的响应的数据，里面可能没有经纬度，后续请做处理。
- 额度用完，重新发起请求
- 地址不对的，去修改地址
"""

import re
import os
import pandas as pd
import asyncio
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import aiohttp
import json
from utils import load_json, save_json

MAX_ERR_NUM = int(1e3)  # 最大报错次数
err_cnt = 0  # API 调用报错统计，避免额度用完，还一直发送请求
API_CALL_CNT = 0  # API调用次数统计
TIMEOUT = 5  # 请求超时时间
address_col_name = "注册地址"

MAX_WORKERS = 50  # 根据API限制调整并发数
limiter = AsyncLimiter(MAX_WORKERS, 1)

# https://restapi.amap.com/v3/geocode/geo?address={}&output=JSON&key={}

# 百度地图接口配置
API_URL = "https://restapi.amap.com/v3/geocode/geo"
AK = "高德地图key"

file = "data/包含有地址的excel文件.xlsx"
print("load excel")
df = pd.read_excel(file)

api_result_dir = "API_results"  # 地址经纬度暂存文件夹
os.makedirs(api_result_dir, exist_ok=True)
base_name = os.path.basename(file).split(".")[0]

# api_result_file of current file
cur_api_result_file = os.path.join(api_result_dir, "cur", f"{base_name}.json")

# 为避免覆盖文件，提前检查输出文件是否存在，若存在请转移到 API_results/pre 文件夹下
assert not os.path.exists(cur_api_result_file)

ANS = {}  # 只和当前这个文件有关，最后导出ANS，导出的文件只包括当前这个excel文件的经纬度信息
PRE_DATA = {}  # 所有的已经爬取过的经纬度数据，包括与当前excel无关的数据

# 加载api_result_dir，文件夹下的所有json文件，合并到一起
print("loading pre data")
for name in os.listdir(p := os.path.join(api_result_dir, "pre")):
    file_name = os.path.join(p, name)
    PRE_DATA.update(load_json(file_name))

# 直接从注册地址列生成地址列表，过滤空值并去除空格
ADDRESSES = df[address_col_name].dropna().str.strip().tolist()
ADDRESSES = [re.sub(r"（.*?）", "", add) for add in ADDRESSES]
# 从 ADDRESSES 中删除PRE_DATA里面已经有的数据
ADDRESSES = [add for add in ADDRESSES if add not in PRE_DATA.keys()]


async def geocoding(session, address, progress):

    # https://restapi.amap.com/v3/geocode/geo?address=北京市朝阳区阜通东大街6号&output=XML&key=<用户的key>
    params = {
        "address": address,
        "output": "JSON",
        "key": AK,
        # "ret_coordtype": "gcj02ll",  # 可根据需要修改坐标系, 国测局坐标
    }
    global err_cnt
    global ANS
    global PRE_DATA
    global API_CALL_CNT

    if address in PRE_DATA.keys():
        progress.update(1)
        ANS.update({address: PRE_DATA[address]})
        return PRE_DATA[address]

    _func_ans = {}

    try:
        async with limiter:
            async with session.get(API_URL, params=params, timeout=TIMEOUT) as response:
                if response.status == 200:
                    data = await response.text()
                    data = json.loads(data)
                    # 添加辅助信息，便于后续数据提取

                    # 删除idx，不再支持index
                    data.update({"address_input": address})

                    if data.get("status") not in [1, "1"]:
                        err_cnt += 1
                        print(data, "status error", [data.get("status")])

                    # 无论服务器的响应是否正确，都保存地址，避免对错误的地址发送多次请求
                    API_CALL_CNT += 1
                    _func_ans = {address: data}
                    progress.update(1)
                else:
                    # 服务器错误,重新发起请求
                    print("response.status error", address)
    except Exception as e:
        err_cnt += 1
        print("geocoding except", address, e.args)

    if _func_ans:
        ANS.update(_func_ans)
        PRE_DATA.update(_func_ans)

    if err_cnt >= MAX_ERR_NUM:
        # 能够被 __main__ 异常捕获住，并完成已有数据保存
        raise Exception(f"Arrive the MAX_ERR_NUM ({str(MAX_ERR_NUM)}), stop running!")

    # API调用达到一定次数后，保存
    if API_CALL_CNT % 4000 == 0:
        save_json(ANS, cur_api_result_file)
    return None


# 运行多个 API 调用并保存结果
async def fetch_all():
    tasks = []
    with tqdm(total=len(ADDRESSES), desc="Processing") as progress:
        async with aiohttp.ClientSession() as session:
            for idx, address in enumerate(ADDRESSES):
                if address is None:
                    continue
                tasks.append(geocoding(session, address, progress))
            await asyncio.gather(*tasks)  # 并发执行所有任务


def amap_export2df(lng_lat_d, source_df, output_file, address_col_name):
    """
    高德地图给excel添加经纬度属性列
    """

    def _get_lon_lat(address):
        d = lng_lat_d.get(address, None)
        lngs, lats = pd.NA, pd.NA
        if d:
            # lngs = d["geocodes"]["location"]["lng"]
            # lats = d["result"]["location"]["lat"]
            geocodes = d.get("geocodes", [])
            if len(geocodes) > 0:
                location = geocodes[0]["location"]
                lngs, lats = location.split(",")
        return lngs, lats

    def _set_lon_lat_on_df(row):
        lngs, lats = _get_lon_lat(row[address_col_name])
        row["经度"] = lngs
        row["纬度"] = lats
        return row

    target_df = source_df.apply(_set_lon_lat_on_df, axis=1)
    target_df.to_excel(output_file, index=False)


if __name__ == "__main__":
    try:
        # # 检查是否在Jupyter环境中运行
        # try:
        #     get_ipython()
        #     # 如果在Jupyter中，使用nest_asyncio
        #     import nest_asyncio
        #
        #     nest_asyncio.apply()
        #     asyncio.run(fetch_all())
        # except NameError:
        #     # 如果不在Jupyter中，直接运行
        #     asyncio.run(fetch_all())
        asyncio.run(fetch_all())
    except Exception as e:
        print("main except", e.args)
    finally:
        save_json(ANS, cur_api_result_file)
