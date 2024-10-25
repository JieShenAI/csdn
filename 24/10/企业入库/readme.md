# 大规模企业工商信息导入MySQL数据库的技术实践

在处理海量企业工商信息时，MySQL数据库是一种可靠的存储方式。本文将介绍如何将超过2亿条企业数据导入MySQL数据库，并讨论相关的技术细节，包括表结构设计、数据预处理和批量插入策略。

## 摘要：
本文介绍了将超过2亿条企业工商信息高效导入MySQL数据库的技术方法，包括表结构设计、数据预处理、批量插入和错误处理等关键步骤。针对大规模数据存储，本文提供了索引优化使用等性能优化建议，确保数据处理的高效性。

---

`code.py`: 为可运行的代码，只需要输入存放工商数据的文件夹路径即可。修改Mysql数据库名和密码。

csdn原始文章：[]()


## 1. 表结构设计

创建数据库：

```sql
CREATE DATABASE industry
CHARACTER SET utf8mb4
COLLATE utf8mb4_general_ci;
```



在设计数据库表时，我们要考虑到数据的特点和存储效率。这里给出的`companies`表结构包含了企业的基本信息字段，如企业名称、统一社会信用代码、注册资本、地址、行业分类等。下面是表的建表SQL语句：

```sql
CREATE TABLE companies (
    id INT AUTO_INCREMENT PRIMARY KEY,
    uuid CHAR(36) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    english_name VARCHAR(255),
    unified_social_credit_code VARCHAR(255),
    company_type VARCHAR(100),
    business_status VARCHAR(255),
    establishment_date DATE,
    approval_date DATE,
    legal_representative VARCHAR(255),
    registered_capital VARCHAR(100),
    paid_in_capital VARCHAR(100),
    insured_number INT,
    company_size VARCHAR(100),
    business_scope VARCHAR(2000),
    registered_address VARCHAR(255),
    business_period VARCHAR(255),
    taxpayer_identification_number VARCHAR(255),
    business_registration_number VARCHAR(255),
    organization_code VARCHAR(255),
    taxpayer_qualification VARCHAR(100),
    former_name VARCHAR(500),
    province VARCHAR(100),
    city VARCHAR(100),
    district VARCHAR(100),
    website_link VARCHAR(1023),
    industry VARCHAR(100),
    primary_industry_category VARCHAR(100),
    secondary_industry_category VARCHAR(100),
    tertiary_industry_category VARCHAR(100),
    registration_authority VARCHAR(255),
    longitude DECIMAL(10, 7),
    latitude DECIMAL(10, 7),
    website VARCHAR(255)
);
```

### 1.1 数据量大的设计注意事项

由于数据量巨大，不建议轻易使用`unique`约束，因为大量数据的唯一性检查会影响写入性能。同时，字段长度应合理设置，例如`VARCHAR`长度，避免占用过多存储空间。

## 2. 数据预处理

数据预处理主要包括字段映射、异常值处理和数据截断。以下是数据处理的Python代码及其步骤：

### 2.1 字段映射

由于原始数据中的字段名为中文，需要将其映射为英文对应的字段名，以便插入数据库。映射关系如下：

```python
attr_eng_map = {
    "企业名称": "company_name",
    "英文名称": "english_name",
    "统一社会信用代码": "unified_social_credit_code",
    ...
}
```

### 2.2 字段截断

为了防止数据插入时超过字段长度导致的错误，对于较长的字段（如`注册地址`、`经营范围`等），在插入前需要进行截断处理。例如：

```python
trunc_item = {
    "注册地址" : 255,
    "经营范围": 2000,
    ...
}
```

### 2.3 异常数据处理

对空值和异常数据进行处理，特别是数值字段，需要转换为合适的类型。`trans2int`函数实现了将字段转换为整数的功能，代码如下：

```python
def trans2int(item):
    if pd.isna(item):
        return None
    try:
        return int(eval(item))
    except:
        return None
```

## 3. 数据批量插入

数据插入过程中采用批量插入的方式，以提高写入效率。每次批量插入1000条数据，当达到批次大小后执行批量插入，减少数据库的频繁访问。

### 3.1 批量插入的SQL语句

通过使用Python的`pymysql`库，构建批量插入的SQL语句：

```python
sql = f"""
    INSERT INTO companies (
        {", ".join(list(attr_eng_map.values()))}
    ) VALUES (
        {', '.join(['%s'] * len(list(attr_eng_map.values())))}
    );
""".strip()
```

### 3.2 批量执行插入

使用`executemany`方法将数据批量插入MySQL数据库：

```python
if len(batch) >= BATCH_SIZE:
    cursor.executemany(sql, batch)
    batch = []  # 清空批次
```

## 4. 错误处理和日志记录

数据插入过程中可能会遇到异常情况，如数据格式不正确或数据库连接失败。为了确保数据的一致性和安全性，使用了事务处理机制：

```python
try:
    with connection.cursor() as cursor:
        ...
        connection.commit()  # 提交事务
except Exception as e:
    print(f"插入数据时出现错误: {e}")
    connection.rollback()  # 回滚事务
finally:
    connection.close()  # 关闭连接
```

此外，在大批量数据插入的过程中，使用`nohup`命令执行脚本并将日志输出到文件中，方便后续调试和数据恢复：

```bash
nohup python init_table.py > init_table.log 2>&1 &
```

## 5. 运行效果
下图展示了插入到 Mysql 数据库中的速度，每秒大约插入1.3 万条数据。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e348219fb1df4fe394f5a0ae739b957d.png)

有2亿多条的企业工商信息需要插入到Mysql数据库中，需要等待几个小时。



下述是数据库中的一部分属性和数据的展示图：

![image-20241025195444937](/Users/jie/Library/Application Support/typora-user-images/image-20241025195444937.png)

## 6. 性能优化建议

- **索引设计**：可以为常用的查询字段建立索引，如`company_name`、`unified_social_credit_code`等。

  比如：

  ```sql
  CREATE INDEX idx_company_name ON companies (company_name);
  ```

- **数据分区**：针对海量数据，可以使用分区表来提高查询性能。

- **批量大小调整**：根据实际情况调整批量插入的大小，避免一次性插入过多数据导致内存溢出。

## 7. 总结

本文介绍了如何将大量企业工商信息高效地导入MySQL数据库，包括表结构设计、数据预处理、批量插入和错误处理等方面的技术细节。这种方法适用于大规模数据的导入和处理，同时能保证数据的安全性和一致性。

希望这些技术点能够帮助开发者更好地管理和存储海量数据，提升数据库的处理能力。

## 8. 附录-代码
下述提供了完整的可运行的代码，`folder`文件夹是存储企业工商信息的文件夹。

下图展示了 `folder` 文件夹中包含的csv表格数据。

![image-20241025195735691](/Users/jie/Library/Application Support/typora-user-images/image-20241025195735691.png)

```python
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
```

