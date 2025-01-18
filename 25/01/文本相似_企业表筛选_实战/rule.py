import evaluate
import pandas as pd
from tqdm import tqdm


TOP_K = 10

google_bleu = evaluate.load("google_bleu")


raw_file = "data/init_values.xls"
ref_file = "data/alter_values.xls"
raw_df = pd.read_excel(raw_file)
ref_df = pd.read_excel(ref_file)


raw_texts = raw_df["行业分类"].to_list()
ref_texts = ref_df["类别名称2017"].to_list()


def get_similarity_score(text1: str, text2: str):
    text1 = " ".join(list(text1))
    text2 = " ".join(list(text2))
    d = google_bleu.compute(predictions=[text1], references=[[text2]])
    return d["google_bleu"]


data = []

for text1 in tqdm(raw_texts):
    arr = []
    for text2 in ref_texts:
        similary_score = get_similarity_score(text1, text2)
        arr.append((text2, similary_score))

    # 分数从高到低排序
    arr.sort(key=lambda x: x[1], reverse=True)
    arr = arr[:TOP_K]

    ans = []
    for text, score in arr:
        if score == 0:
            ans.append(pd.NA)
        else:
            ans.append(text)

    data.append(ans)


tmp_df = pd.DataFrame(data, columns=list(range(TOP_K)))
ans_df = pd.concat([raw_df, tmp_df], axis=1)
ans_df.to_csv("规则筛选.csv", index=False)
