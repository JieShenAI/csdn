import re
import logging
from typing import List
import os
import pandas as pd
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from .mongo_db import MapMongoDB
from .argument import ExcelFileArguments


def remove_parentheses_from_address(address: str) -> str:
    """
    删除地址中，中文括号包裹的内容
    """
    return re.sub(r"（.*?）", "", address)


@dataclass
class ExcelObj:
    data_args: ExcelFileArguments

    def __post_init__(self):
        basename = os.path.basename(self.data_args.filename)
        name = basename.split(".")[0]
        self.output_filename = os.path.join(self.data_args.output_dir, f"{name}.{self.data_args.output_type}")
        if not self.data_args.overwrite and os.path.exists(self.output_filename):
            raise ValueError(
                "{self.output_filename} have already existed, but overwrite is False."
            )
        self.read_file_func = pd.read_csv
        if self.data_args.filename.endswith(".xls") or self.data_args.filename.endswith(".xlsx"):
            self.read_file_func = pd.read_excel

    @property
    def address(self) -> List[str]:
        logger.info(f"loadding {self.data_args.filename}")
        df = self.read_file_func(
            self.data_args.filename, usecols=[self.data_args.address_col_name]
        ).dropna()

        if self.data_args.address_clean:
            df[self.data_args.address_col_name] = df[self.data_args.address_col_name].map(
                remove_parentheses_from_address
            )
        address_data = df[self.data_args.address_col_name].to_list()
        return address_data


if __name__ == "__main__":
    address = "123（测试）456"
    address = remove_parentheses_from_address(address)
    print(address)
