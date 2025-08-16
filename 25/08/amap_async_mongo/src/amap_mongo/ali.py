import os
from transformers import HfArgumentParser
import pandas as pd
import asyncio
import logging

from .argument import MongoArguments, AsyncMapCallArguments, ExcelFileArguments
from .mongo_db import MapMongoDB
from .sync import AsyncMapCall
from .data import ExcelObj

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def insert2mongo():
    """
    Asynchronously call the map API and then upload the data to MongoDB.
    """
    logger.info("insert2mongo start ...")
    parser = HfArgumentParser((MongoArguments, AsyncMapCallArguments, ExcelFileArguments))  # type: ignore
    mongo_args, async_args, data_args = parser.parse_args_into_dataclasses()
    mongo_args: MongoArguments
    async_args: AsyncMapCallArguments
    data_args: ExcelFileArguments
    mongo_db = MapMongoDB(mongo_args=mongo_args)
    sync_call = AsyncMapCall(mongo_db, async_args=async_args)
    excel_obj = ExcelObj(data_args=data_args)

    asyncio.run(
        sync_call.insert2mongo(
            excel_obj.address, address_min_length=mongo_args.address_min_length
        )
    )


def export2excel():
    """
    Query the longitude and latitude from MongoDB based on the addresses in the Excel file.
    """
    """
        Asynchronously call the map API and then upload the data to MongoDB.
    """

    logger.info("export2excel start ...")
    parser = HfArgumentParser((MongoArguments, ExcelFileArguments))  # type: ignore
    mongo_args, data_args = parser.parse_args_into_dataclasses()
    mongo_args: MongoArguments
    data_args: ExcelFileArguments
    mongo_db = MapMongoDB(mongo_args=mongo_args)
    excel_obj = ExcelObj(data_args=data_args)

    logger.info(f"loading {data_args.filename} total file ...")
    if data_args.filename.endswith(".xlsx") or data_args.filename.endswith(".xls"):
        df = pd.read_excel(data_args.filename)
    elif data_args.filename.endswith(".csv"):
        df = pd.read_csv(data_args.filename, low_memory=False)
    else:
        raise ValueError(f"{data_args.filename} is not a .csv 、.xlsx or .xls file")

    lon_lat_data = []
    for address in df[data_args.address_col_name]:
        longs, lats = pd.NA, pd.NA
        if len(address) > mongo_args.address_min_length:
            d = mongo_db.query_by_address(address)
            if d:
                geocodes = d.get("geocodes", [])
                if len(geocodes) > 0:
                    location = geocodes[0]["location"]
                    longs, lats = location.split(",")

        lon_lat_data.append((longs, lats))

    df["经度"] = [item[0] for item in lon_lat_data]
    df["维度"] = [item[1] for item in lon_lat_data]
    os.makedirs(data_args.output_dir, exist_ok=True)
    if data_args.output_type == "csv":
        df.to_excel(excel_obj.output_filename, index=False)
