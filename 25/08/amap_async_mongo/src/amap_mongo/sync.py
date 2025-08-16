import os
import json
import asyncio
from typing import List
from aiolimiter import AsyncLimiter
from tqdm import tqdm
import aiohttp
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

from .argument import AsyncMapCallArguments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .mongo_db import MapMongoDB


@dataclass
class AsyncMapCall:
    mongo_db: MapMongoDB
    async_args: AsyncMapCallArguments

    def __post_init__(self):
        self.limiter = AsyncLimiter(max_rate=self.async_args.limiter_ratio, time_period=1)

        if not self.async_args.api_key:
            try:
                load_dotenv()
            except FileNotFoundError:
                raise ValueError("未找到 .env 文件，请创建并设置 api_key")

            api_key = os.getenv("api_key", "")
            if not api_key:
                raise ValueError("在 .env 文件中未找到 api_key，请设置 api_key")

            self.api_key = api_key

    async def __call_api(self, session, address, progress, return_data=False):
        # https://restapi.amap.com/v3/geocode/geo?address=北京市朝阳区阜通东大街6号&output=JSON&key=<用户的key>
        params = {
            "address": address,
            "output": "JSON",
            "key": self.api_key,
        }

        # Above all，search address in mongoDB
        if return_data:
            mongo_db_data = self.mongo_db.query_by_address(address)
            if mongo_db_data:
                progress.update(1)
                return mongo_db_data
        else:
            is_exists = self.mongo_db.is_address_exists(address)
            if is_exists:
                progress.update(1)
                return

        data = {}
        try:
            async with self.limiter:
                async with session.get(
                        self.async_args.base_url, params=params, timeout=self.async_args.timeout
                ) as response:
                    if response.status == 200:
                        data = await response.text()
                        data = json.loads(data)

                        # 删除idx，不再支持index
                        data.update({"address_input": address})
                        # 无论服务器的响应是否正确，都保存地址，避免对错误的地址发送多次请求
                        self.mongo_db.insert_data(**data)
                        progress.update(1)
                    else:
                        # server error
                        print(f"response.status {response.status}", address)
        except Exception as e:
            print("call_api", address, e.args)

        if return_data:
            return data
        else:
            return

    # 运行多个 API 调用并保存结果
    async def insert2mongo(self, addresses: List[str], address_min_length):
        """
        由于协程采取了限速的操作，得先筛选出mongo_db中不存在的地址，再针对mongo_db不存在的地址调用API。
        否则查询mongo_db也会被限速
        """
        logger.info("Check which addresses already exist in the database ...")
        pre_length = len(addresses)

        addresses = [
            address for address in addresses if len(address) > address_min_length
        ]

        addresses = [
            address
            for address in addresses
            if not self.mongo_db.is_address_exists(address)
        ]

        new_length = len(addresses)
        logger.info(
            f"Addresses: {pre_length} total, {pre_length - new_length} in DB, {new_length} to process."
        )
        addresses = addresses[: self.async_args.max_addresses_num]
        logging.info(
            f"max_addresses_num is {self.async_args.max_addresses_num}. Processing {len(addresses)} of {new_length} available samples."
        )
        tasks = []
        with tqdm(total=len(addresses), desc="Processing") as progress:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.__call_api(session, address, progress, return_data=False)
                    for address in addresses
                ]
                # 并发执行所有任务
                await asyncio.gather(*tasks)
