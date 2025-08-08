import re
import os
import pandas as pd
import asyncio
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import aiohttp
import json
from utils import load_json, save_json


api_result_dir = "API_results"  # 地址经纬度暂存文件夹
PRE_DATA = {}  # 是一个大库
# 加载api_result_dir，文件夹下的所有json文件，合并到一起
# 所以最后导出的json文件，包含了之前所有的数据

print("loading pre data")
for name in os.listdir(p := os.path.join(api_result_dir, "pre")):
    file_name = os.path.join(p, name)
    PRE_DATA.update(load_json(file_name))


file = "data/jiekai_industry.xlsx"
print("load excel")

df = pd.read_excel(file)


def amap_export2df(pre_data, source_df, output_file, address_col_name):
    """
    高德地图给excel添加经纬度属性列
    """

    def _get_lon_lat(address):
        d = pre_data.get(address, None)
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
    return target_df


output_df = amap_export2df(PRE_DATA, df, "output/文件夹名.xlsx", "注册地址")

# output_df = pd.read_excel(
#     "output/jiekai_industry.xlsx",
#     usecols=["注册地址", "经度", "纬度"],
#     low_memory=False,
# )

new_df = output_df.dropna(subset="纬度")
print("有效地址的shape:", new_df.shape)
