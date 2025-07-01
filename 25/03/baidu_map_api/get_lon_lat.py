"""
协程调用百度地图API，获取对应的经纬度
1. 从 api_result_dir 文件夹加载之前已经保存的文件
2. 新保存到api_result_dir的json文件，包含之前所有的API调用结果
"""

import os
import pandas as pd
import asyncio
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import aiohttp
import json
from utils import save_json, load_json

MAX_ERR_NUM = 100  # 最大报错次数
err_cnt = 0  # API 调用报错统计，避免额度用完，还一直发送请求
API_CALL_CNT = 0  # 百度网盘API调用次数统计
TIMEOUT = 5  # 请求超时时间
address_col_name = "注册地址"

MAX_WORKERS = 2.8  # 根据API限制调整并发数
limiter = AsyncLimiter(MAX_WORKERS, 1)

# 百度地图接口配置
API_URL = "https://api.map.baidu.com/geocoding/v3"
AK = "百度地图的KEY"

file = "excel的路径.xlsx"
df = pd.read_excel(file)

api_result_dir = "API_results"  # 地址经纬度暂存文件夹
os.makedirs(api_result_dir, exist_ok=True)
base_name = os.path.basename(file).split(".")[0]

# api_result_file of current file
cur_api_result_file = os.path.join(api_result_dir, f"{base_name}.json")
ANS = {}
PRE_DATA = {}
# 加载api_result_dir，文件夹下的所有json文件，合并到一起
# 所以最后导出的json文件，包含了之前所有的数据
for name in os.listdir(api_result_dir):
    file_name = os.path.join(api_result_dir, name)
    PRE_DATA.update(load_json(file_name))

# 直接从注册地址列生成地址列表，过滤空值并去除空格
ADDRESSES = df[address_col_name].dropna().str.strip().tolist()


async def geocoding(session, address, progress):
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

                    if data.get("status") == 0:
                        progress.update(1)
                        _func_ans = {address: data}
                        API_CALL_CNT += 1
                    else:
                        err_cnt += 1
                        print(data)
                else:
                    # 服务器错误,重新发起请求
                    print("response.status error", address)
    except Exception as e:
        err_cnt += 1
        print(address, e.args)

    if _func_ans:
        ANS.update(_func_ans)

    if err_cnt >= MAX_ERR_NUM:
        # save_json(ANS, cur_api_result_file)
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


def export2df(lng_lat_d, source_df, output_file, address_col_name):
    """
    给excel添加经纬度属性列
    """

    def _get_lon_lat(address):
        d = lng_lat_d.get(address, None)
        lngs, lats = pd.NA, pd.NA
        if d:
            lngs = d["result"]["location"]["lng"]
            lats = d["result"]["location"]["lat"]
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
        print(e.args)
    finally:
        save_json(ANS, cur_api_result_file)

    # 导出excel
    output_file = f"{base_name}_经纬度_百度API.xlsx"
    export2df(
        lng_lat_d=ANS,
        source_df=df,
        output_file=output_file,
        address_col_name=address_col_name,
    )
