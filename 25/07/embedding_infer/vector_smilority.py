import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from FlagEmbedding import BGEM3FlagModel

print("load BAAI/bge-m3 ...")
# Setting use_fp16 to True speeds up computation with a slight performance degradation
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,
)


def encode_df(df, output_file):
    embeddings_1 = model.encode(
        df["经营范围"].apply(lambda x: x if isinstance(x, str) else "").to_list(),
        batch_size=64,
        max_length=512,
    )

    embeddings_2 = model.encode(
        df["战略性新兴产业分类名称"]
        .apply(lambda x: x if isinstance(x, str) else "")
        .to_list(),
        batch_size=64,
        max_length=512,
    )
    t1 = embeddings_1["dense_vecs"]
    t2 = embeddings_2["dense_vecs"]
    t1 = t1[:, np.newaxis, :]
    t2 = t2[:, :, np.newaxis]

    # to torch cuda
    t1 = torch.from_numpy(t1).to("cuda")
    t2 = torch.from_numpy(t2).to("cuda")

    scores = t1 @ t2
    scores = scores.flatten().tolist()
    df["经营范围与类别向量相似度"] = scores
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    raw_data_dir = r"C:\Users\1\Desktop\cache\战新企业demo"
    output_dir = r"C:\Users\1\Desktop\cache\vec_similar_industry"

    for sub_fold in os.listdir(raw_data_dir):
        p = os.path.join(raw_data_dir, sub_fold)
        for name in tqdm(os.listdir(p)):
            file_name = os.path.join(p, name)
            df = pd.read_csv(file_name, low_memory=False)
            output_sub_dir = os.path.join(output_dir, sub_fold)
            os.makedirs(output_sub_dir, exist_ok=True)
            output_file = os.path.join(output_sub_dir, name)
            encode_df(df, output_file)
