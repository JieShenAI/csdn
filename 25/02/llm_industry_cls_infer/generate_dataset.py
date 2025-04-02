import pandas as pd
import random
from typing import List
import json
from copy import deepcopy
from tqdm import tqdm


INDUSTRY_NAME = "产业链名"
INDUSTRY_NODE = "产业链环节"
INDUSTRY_DESC = "产业链环节描述"

# df = pd.read_excel("五群行业分类.xlsx")
# df = df.rename(columns={"六链": INDUSTRY_NAME})

df = pd.read_excel("五群行业分类.xlsx", sheet_name="五群")  # 读取所有 Sheet
df = df.rename(columns={"五群": INDUSTRY_NAME})




industry_cls_df = pd.read_excel("小类注释_含一二三位数代码及名称.xlsx")
industry_cls_df.head()


industry_cls_df = industry_cls_df[["小类代码2017", "小类名称2017"]]
industry_cls_df = industry_cls_df.rename(
    columns={"小类代码2017": "code", "小类名称2017": "name"}
)
industry_cls_df["code"] = industry_cls_df["code"].map(lambda x: str(x).zfill(4))


ALL_CLS_NAME = industry_cls_df["name"].values.tolist()


# ## build infer dataset


prompt_template = """
你是资深的产业专家！请先浏览下述产业环节。
{name} {node}产业链环节。
接下来，请你逐一浏览行业分类列表：{selected_elements}，并在其中找出一眼看上去就属于{name} {node}产业链环节的行业，不需要做过渡的行业内容挖掘。
以python列表的格式输出，若没有密切相关的行业就返回空列表[]。输出样例：["xxx", "xxx"]
""".strip()


def select_cls(arr: List, n: int) -> List:
    """
    从 arr 中，随机筛选出 n 个元素，然后从cls中删除筛选出的元素
    """
    n = min(len(arr), n)
    if n == len(arr):
        return arr, []

    # 随机选择 n 个元素
    selected_elements = random.sample(arr, n)

    # 从 arr 中删除选中的元素
    for element in selected_elements:
        arr.remove(element)

    return selected_elements, arr


ALL_CLS_NAME = industry_cls_df["name"].values.tolist()
prompts = []

block_size = 5
for _, row in tqdm(df.iterrows()):
    new_CLS = deepcopy(ALL_CLS_NAME)
    while len(new_CLS) > 0:
        name, node, desc = row[INDUSTRY_NAME], row[INDUSTRY_NODE], row[INDUSTRY_DESC]
        selected_elements, new_CLS = select_cls(new_CLS, block_size)
        prompt = prompt_template.format(
            selected_elements=selected_elements,
            name=name,
            node=node,
            # desc=desc
        )
        prompts.append({"instruction": prompt, "input": "", "output": ""})


def save_json(obj, file):
    with open(file, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


save_json(prompts, "data/industry_cls.json")
print("dataset have generated!")
