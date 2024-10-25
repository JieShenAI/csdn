import os
from tqdm import tqdm
import pandas as pd
import pymysql


attr_eng_map = {
    "企业名称": "company_name",
    "英文名称": "english_name",
    "统一社会信用代码": "unified_social_credit_code",
    "企业类型": "company_type",
    "经营状态": "business_status",
    "成立日期": "establishment_date",
    "核准日期": "approval_date",
    "法定代表人": "legal_representative",
    "注册资本": "registered_capital",
    "实缴资本": "paid_in_capital",
    "参保人数": "insured_number",
    "公司规模": "company_size",
    "经营范围": "business_scope",
    "注册地址": "registered_address",
    "营业期限": "business_period",
    "纳税人识别号": "taxpayer_identification_number",
    "工商注册号": "business_registration_number",
    "组织机构代码": "organization_code",
    "纳税人资质": "taxpayer_qualification",
    "曾用名": "former_name",
    "所属省份": "province",
    "所属城市": "city",
    "所属区县": "district",
    "网站链接": "website_link",
    "所属行业": "industry",
    "一级行业分类": "primary_industry_category",
    "二级行业分类": "secondary_industry_category",
    "三级行业分类": "tertiary_industry_category",
    "登记机关": "registration_authority",
    "经度": "longitude",
    "纬度": "latitude",
    "网址": "website",
}


def trans2int(item):
    if pd.isna(item):
        return None
    try:
        return int(eval(item))
    except:
        return None


def parse_item(row, attr_name):

    # 字符截断
    trunc_item = {
        "注册地址" : 255,
        "网址": 255,
        "网站链接": 255,
        "曾用名": 500,
        "经营范围": 2000,
    }

    ans = []
    for attr in attr_name:
        item = row.get(attr, None)

        if pd.isna(item):
            ans.append(None)
            continue
        elif attr == "参保人数":
            item = trans2int(item)
    
        # 异常字符捕获
        if isinstance(item, str):
            item = item.strip()
            # 有 空串 和 -
            if len(item) in [0, 1]:
                item = None
        
        if attr in trunc_item.keys():
            if isinstance(item, str):
                max_len = trunc_item[attr]
                item = item[:max_len]

        ans.append(item)

    return tuple(ans)


def insert_sql_by_csv(file):
    df = pd.read_csv(file, low_memory=False)

    # 连接到MySQL数据库
    connection = pymysql.connect(
        host="localhost",  # MySQL数据库的主机
        user="root",  # MySQL用户名
        password="密码",  # MySQL密码
        database="数据库名",  # 你要插入数据的数据库
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    
    BATCH_SIZE = 1000

    sql = f"""
            INSERT INTO companies (
                {", ".join(list(attr_eng_map.values()))}
            ) VALUES (
                {', '.join(['%s'] * len(list(attr_eng_map.values())))}
            );
            """.strip()
    
    # 插入数据到MySQL
    try:
        with connection.cursor() as cursor:
            batch = []
            for _, row in tqdm(df.iterrows(), total=len(df)):
                # 企业名称如果是 None 跳过
                company_name = row["企业名称"]
                if company_name is None or not isinstance(company_name, str):
                    continue
                
                batch.append(
                        parse_item(row, list(attr_eng_map.keys()))
                    )
                                
                # 当批次达到 BATCH_SIZE 时执行批量插入
                if len(batch) >= BATCH_SIZE:
                    cursor.executemany(sql, batch)
                    batch = []  # 清空批次
                    
            # 插入剩余的未满批次的数据
            if batch:
                cursor.executemany(sql, batch)

            # 提交事务
            connection.commit()
            
    except Exception as e:
        print(f"插入数据时出现错误: {e}")
        connection.rollback()

    finally:
        connection.close()


if __name__ == "__main__":
    folder = "/home/jie/Desktop/industry_info"
    print(f"文件总数: {len(os.listdir(folder))}")
    cnt = 0
    for file_name in os.listdir(folder):
        if file_name.endswith(".csv"):
            cnt += 1
            filename = os.path.join(folder, file_name)
            print(cnt, file_name)
            insert_sql_by_csv(filename)

    # mysql -h 127.0.0.1 -P 3306 -u root -p
    # nohup python init_table.py > init_table.log 2>&1 &