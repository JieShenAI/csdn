import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 1. 准备
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = (
    AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
)


class PairDataset(Dataset):
    def __init__(self, df):
        self.A = (
            df["经营范围"].apply(lambda x: x if isinstance(x, str) else "").to_list()
        )
        self.B = (
            df["战略性新兴产业分类名称"]
            .apply(lambda x: x if isinstance(x, str) else "")
            .to_list()
        )

    def __len__(self):
        return len(self.A)

    def __getitem__(self, idx):
        return self.A[idx], self.B[idx]


def collate_fn(batch):
    text1s, text2s = zip(*batch)
    enc_A = tokenizer(
        list(text1s), padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    enc_B = tokenizer(
        list(text2s), padding=True, truncation=True, max_length=256, return_tensors="pt"
    )
    return enc_A, enc_B

# tokenizer 放在 collate_fn 里面，使用 max_workers 反而是增加了进程加载与切换的负担
# 尽量让 collate_fn 尽可能轻量（避免计算、日志），故在该脚本中不设置 max_workers

def safe_num_workers(dataset_len, max_workers=4):
    return 0 if dataset_len <= 10000 else max_workers

def run(df, output_file):
    # 2. DataLoader
    dataset = PairDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=64,
        # num_workers=safe_num_workers(len(dataset)),
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    results = []
    for enc_A, enc_B in tqdm(loader):
        # 3. GPU 推理
        for t in (enc_A, enc_B):
            for k, v in t.items():
                t[k] = v.to("cuda", non_blocking=True)
        with torch.no_grad():
            emb_A = model(**enc_A).last_hidden_state[:, 0]
            emb_B = model(**enc_B).last_hidden_state[:, 0]

            emb_A = F.normalize(emb_A, dim=1)
            emb_B = F.normalize(emb_B, dim=1)

        emb_A = emb_A.unsqueeze(1)
        emb_B = emb_B.unsqueeze(1)

        sims = emb_A @ emb_B.transpose(-2, -1)
        sims = sims.flatten().cpu().tolist()
        results.extend(sims)

    # 5. 把 results 写回原表
    df["sim"] = results
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    raw_data_dir = r"C:\Users\1\Desktop\cache\战新企业demo"
    output_dir = r"C:\Users\1\Desktop\cache\new_vec_similar_industry"

    for sub_fold in os.listdir(raw_data_dir):
        p = os.path.join(raw_data_dir, sub_fold)
        for name in os.listdir(p):
            file_name = os.path.join(p, name)
            df = pd.read_csv(file_name, low_memory=False)
            output_sub_dir = os.path.join(output_dir, sub_fold)
            os.makedirs(output_sub_dir, exist_ok=True)
            output_file = os.path.join(output_sub_dir, name)
            run(df, output_file)
