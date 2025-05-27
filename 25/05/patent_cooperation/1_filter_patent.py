import os
import pandas as pd
from tqdm import tqdm

industry_home = "/mnt/mydisk/pku_data/3571万专利申请全量数据1985-2022年/"

# 加载目标企业名单


target_company = pd.read_csv("data/2w_企业名称.csv")

target_companies = set(target_company["企业名称"].values)


def has_target_company(applicants: str) -> bool:
    names = applicants.split('; ')
    names = list(filter(lambda x:len(x) >= 4, names))
    if len(names) <= 1:
        return False
    return any(name in target_companies for name in names)


output_folder = "output/provs"

for file in tqdm(os.listdir(industry_home)):
    output_file = os.path.join(output_folder, file)
    if not file.endswith(".csv") or os.path.exists(output_file):
        continue
    file_name = os.path.join(industry_home, file)
    tmp_df = pd.read_csv(file_name, low_memory=False)
    tmp_df = tmp_df[tmp_df['申请人'].str.contains(';', na=False)]
    tmp_df = tmp_df[tmp_df['申请人'].apply(has_target_company)]
    tmp_df.to_csv(output_file, index=False)
    print(file, len(tmp_df))


# 使用把output文件夹里面所有的文本concat到一起

