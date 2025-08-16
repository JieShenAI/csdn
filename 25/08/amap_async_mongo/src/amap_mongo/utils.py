import json


def save_json(obj: dict, filename: str):
    """
    将字典对象保存为 JSON 文件。

    参数:
    obj (dict): 要保存的字典对象。
    filename (str): 保存的文件名，默认为 "data.json"。
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=4)


def load_json(filename: str) -> dict:
    """
    从 JSON 文件中加载数据并返回字典对象。

    参数:
    filename (str): 要加载的 JSON 文件名

    返回:
    dict: 从 JSON 文件中加载的字典对象。如果加载失败，则返回 None。
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data
