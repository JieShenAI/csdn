"""
该脚本用于从谷歌学术中搜索论文信息，并保存到CSV文件中。
读取与输出都是同一个csv文件， 通过 source_df.to_dict(orient="records") 获取原文件的 List[Dict] 数据。

scholar 爬虫有一些还不足的地方，运行次数过多后，可能需要验证码、封ip的问题。
下述代码的逻辑是，访问首页输入关键词，再搜索论文。其实不需要每次都访问首页，这会造成访问次数过多。
driver.get("https://scholar.google.com/")
"""

# !pip install selenium undetected-chromedriver beautifulsoup4

import time
import json
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


def start_driver(proxy=None):
    options = uc.ChromeOptions()
    # options.add_argument("--headless")  # 不打开窗口
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    options.add_argument("user-agent=Mozilla/5.0")

    # 设置本地代理（如果需要）
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")

    driver = uc.Chrome(options=options)
    return driver


tmp_file = "tmp.csv"


def search_scholar(driver, query, max_results=10):

    driver.get("https://scholar.google.com/")
    time.sleep(2)

    # 输入关键词并搜索
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(3)

    # 解析页面
    soup = BeautifulSoup(driver.page_source, "html.parser")
    # results = []
    if soup.select(".gs_r.gs_or") is None:
        return {}
    
    # for 循环停止，只获取第一个元素
    for entry in soup.select(".gs_r.gs_or")[:max_results]:
        title_tag = entry.select_one(".gs_rt")
        if title_tag:
            title = title_tag.get_text()
            link = title_tag.a["href"] if title_tag.a else None
        else:
            title = "N/A"
            link = None

        author_info = entry.select_one(".gs_a")
        snippet = entry.select_one(".gs_rs")
        cite_tag = entry.select_one(".gs_fl a:contains('Cited by')")
        cited_by = 0
        if cite_tag:
            try:
                cited_by = int(cite_tag.get_text().split()[-1])
            except:
                cited_by = 0

        # results.append()
        scholar_info = {
            "title": title,
            "link": link,
            "authors": author_info.get_text() if author_info else "N/A",
            "snippet": snippet.get_text() if snippet else "N/A",
            "cited_by": cited_by,
        }

        return scholar_info


def equal(a: str, b: str):
    a = a.strip().lower()
    b = b.strip().lower()

    def _replace(text: str):
        replace_chs = ["-", ":", "："]
        for ch in replace_chs:
            text = text.replace(ch, "")
        return text

    a = _replace(a)
    b = _replace(b)
    return a == b


# 示例调用
if __name__ == "__main__":
    proxy = "http://127.0.0.1:7890"  # 可选：设置为 None 表示不使用代理
    driver = start_driver(proxy)
    source_filename = "source.csv"
    source_df = pd.read_csv(source_filename)
    new_source_data = source_df.to_dict(orient="records")
    try:
        for idx, row in tqdm(source_df.iterrows()):
            row_d = row.to_dict()
            paper_name = row["paper_name"].strip()
            scholar_info = row.get("scholar_info", pd.NA)
            if not pd.isna(scholar_info):
                continue
            scholar_info: dict = search_scholar(driver, paper_name, max_results=5)

            new_source_data[idx].update(
                {
                    "valid": equal(scholar_info["title"], paper_name),
                    "scholar_info": json.dumps(scholar_info),
                }
            )
            # for i, paper in enumerate(papers, 1):
            #     print(f"\n[{i}] {paper['title']}")
            #     print(f"链接: {paper['link']}")
            #     print(f"作者与出版信息: {paper['authors']}")
            #     print(f"摘要: {paper['snippet']}")
            #     print(f"引用次数: {paper['cited_by']}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
        new_source_df = pd.DataFrame(new_source_data)
        new_source_df.to_csv("source.csv", index=False)
        
