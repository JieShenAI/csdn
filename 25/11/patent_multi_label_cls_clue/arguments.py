from dataclasses import dataclass
from typing import Union

@dataclass
class PatentModelArgs:
    model_name_or_path: str


@dataclass
class PatentDataArgs:
    patent_train_json_file: str = ""
    text_max_length: int = 512
    patent_predict_csv_file: str = ""
    train_dataset_size_or_ratio: float = 0.8
    eval_dataset_size_or_ratio: float = 0.2