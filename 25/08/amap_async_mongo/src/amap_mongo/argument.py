import os
from dataclasses import dataclass, field


@dataclass
class MongoArguments:
    db_name: str
    collection_name: str
    mongo_uri: str = field(default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    address_min_length: int = field(default=5)


@dataclass
class AsyncMapCallArguments:
    limiter_ratio: float = field(default=2.8)
    timeout: float = field(default=5.0)
    base_url: str = field(default="https://restapi.amap.com/v3/geocode/geo")
    api_key: str = field(default="")
    max_addresses_num: int = field(
        default=5000,
        metadata={"help": "地址解析的最大调用数量，默认是免费用户的5k次数"},
    )


@dataclass
class ExcelFileArguments:
    """
    Only support file type in [.csv, .xls, xlsx]
    """

    filename: str
    address_col_name: str
    overwrite: bool = field(default=False)
    output_dir: str = field(default="output")
    address_clean: bool = field(default=False, metadata={"help": "是否对地址进行处理"})
    output_type: str = field(default="csv", metadata={"help": "csv or xlsx"})
