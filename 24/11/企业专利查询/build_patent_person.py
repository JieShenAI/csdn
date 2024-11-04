import os
import re
import pymysql
import pandas as pd
from tqdm import tqdm

PASSWORD = "数据库密码"
DATABASE = "数据库名"

# 专利字段映射
Patent_Table_Column = {
    "申请人": "applicant",
    "专利公开号": "publication_number",
    "申请日": "application_date",
    "申请公布日": "publication_date",
    "授权公布日": "grant_publication_date",
}


def filter_company(applicant):
    """
    提取中文公司名称，并去除空格
    """
    if applicant is None or not isinstance(applicant, str):
        return []

    split_pattern = r"[;；]"
    applicant = re.split(split_pattern, applicant)
    applicant = map(str.strip, applicant)
    return list(filter(lambda x: len(x) >= 4, applicant))


def insert_sql_by_csv(file_name):
    df = pd.read_csv(file_name, low_memory=False)
    BATCH_SIZE = 3000
    table_column_en = list(Patent_Table_Column.values())

    # 连接到MySQL数据库
    connection = pymysql.connect(
        host="localhost",  # MySQL数据库的主机
        user="root",  # MySQL用户名
        password=PASSWORD,  # MySQL密码
        database=DATABASE,  # 你要插入数据的数据库
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with connection.cursor() as cursor:
            sql = f"""
                    INSERT INTO patent_p ({", ".join(table_column_en)}, score) 
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """.strip()
            batch_data = []

            for _, row in tqdm(df.iterrows(), total=len(df)):
                d = {}
                applicants = []

                for zh_k, en_k in Patent_Table_Column.items():
                    item = row[zh_k]
                    if pd.isna(item):
                        item = None

                    if zh_k == "申请人":
                        applicants = filter_company(item)
                    else:
                        d[en_k] = item

                for pos, applicant in enumerate(applicants):
                    d["applicant"] = applicant
                    d["score"] = 1 / (pos + 1)

                    tmp_values = tuple([d[k] for k in table_column_en + ["score"]])
                    batch_data.append(tmp_values)
                    if len(batch_data) >= BATCH_SIZE:
                        cursor.executemany(sql, batch_data)
                        # 清空批次
                        batch_data = []

            if batch_data:
                cursor.executemany(sql, batch_data)
            connection.commit()

    except Exception as e:
        print(f"插入数据时出现错误: {e}")
        connection.rollback()
    finally:
        connection.close()


if __name__ == "__main__":

    folder = "/xxx/3571万专利申请全量数据1985-2022年/"
    print(f"文件总数: {len(os.listdir(folder))}")
    cnt = 0
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            cnt += 1
            filename = os.path.join(folder, file_name)
            print(cnt, file_name)
            insert_sql_by_csv(filename)
