import json
from typing import Dict


def save_json(data: Dict, file: str):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(file) -> Dict:
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)
