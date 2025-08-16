import os
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from dataclasses import dataclass, field

from .argument import MongoArguments


@dataclass
class MapMongoDB:
    mongo_args: MongoArguments

    def insert_data(self, **kwargs):
        try:
            # 插入文档
            self.collection.insert_one(kwargs)
            return 1
        except DuplicateKeyError:
            return 0
        except Exception as e:
            return -1

    # query item
    def query_by_address(self, address):
        result = self.collection.find_one({"address_input": address})
        return result

    def is_address_exists(self, address) -> bool:
        result = self.collection.find_one({"address_input": address})
        if result is None:
            return False
        return True

    def __post_init__(self):
        """初始化数据库，创建集合并为 name 字段添加唯一索引"""
        # 连接 MongoDB
        self.client = MongoClient(self.mongo_args.mongo_uri)
        self.db = self.client[self.mongo_args.db_name]
        self.collection = self.db[self.mongo_args.collection_name]
        self.collection.create_index("address_input", unique=True)

    def __del__(self):
        self.client.close()
