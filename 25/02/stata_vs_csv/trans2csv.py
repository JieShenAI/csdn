import os
import pandas as pd
from tqdm import tqdm


folder = r"E:\BaiduNetdiskDownload\1949～2023年工商企业注册信息（V2025）\unzip_分年"
csv_home = r"E:\BaiduNetdiskDownload\1949～2023年工商企业注册信息（V2025）\csv_data"
files = os.listdir(folder)

# 读取耗时7分钟
for idx, file in tqdm(enumerate(files)):
    raw_file = os.path.join(folder, file)
    output_file = os.path.join(csv_home, f"{os.path.basename(file).split('.')[0]}.csv")

    if os.path.exists(output_file):
        print(f"{idx} - {file} already existed, just skip!")
        continue
    
    print(f"{idx} - {file} loading ...")
    df = pd.read_stata(raw_file)
    print("writing csv, please wait!")
    print()
    df.to_csv(output_file, index=False)
