import os
import pandas as pd
import pymysql
# import argparse

database = "数据库名"
password = "数据库密码"


connection = pymysql.connect(
    host="localhost",  # MySQL数据库的主机
    user="root",  # MySQL用户名
    password=password,  # MySQL密码
    database=database,  # 插入数据的数据库
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)

columns = list(range(1985, 2024)) + ["专利件数", "专利得分"]


def get_patent_statistics_by_name(name):
    if not name:
        return {}

    sql = f"""select applicant as company_name, YEAR(application_date) as year, count(*) as cnt, sum(score) from patent_p 
    where applicant='{name}'
    group by YEAR(application_date);
    """
    with connection.cursor() as cursor:
        data = cursor.execute(sql)
        data = cursor.fetchall()

    ans = {}
    cnt = 0
    score = 0

    for k in columns:
        ans[k] = None

    for item in data:
        cnt += item.get("cnt", 0)
        score += item.get("sum(score)", 0)

        year = item.get("year", None)
        if year:
            ans[year] = item.get("cnt", 0)

    ans["专利得分"] = score
    ans["专利件数"] = cnt
    return pd.Series(ans)


def add_patent_data(input_file, company_name_field="企业名称"):

    print("open", input_file)
    # 读取 CSV 文件
    df = pd.read_csv(input_file, low_memory=False)
    
    df[columns] = df[company_name_field].apply(get_patent_statistics_by_name)

    folder_path = os.path.dirname(input_file)
    output_file = os.path.basename(input_file).split(".")[0] + "_专利统计.xlsx"
    # 保存更新后的数据到 CSV 文件
    output_file = os.path.join(folder_path, output_file)
    
    df.to_excel(output_file, index=False)
    print(f"专利数据已成功添加到文件：{output_file}")


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description="Add patent counts to industry.csv")
    # parser.add_argument("input_file", help="The input CSV file with industry data")
    # parser.add_argument(
    #     "-name", "--name", default="企业名称", help="The column name for company names"
    # )
    # args = parser.parse_args()

    # # 调用函数处理文件
    # add_patent_data(args.input_file, args.name)
    
    folder = "/.../pku_industry/csv_folder_test"
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        file_name = os.path.join(folder, file)
        add_patent_data(file_name)

    connection.close()
