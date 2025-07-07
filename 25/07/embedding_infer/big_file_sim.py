import os
import glob
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download("BAAI/bge-m3")

# 1. 准备
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).cuda().eval()


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

# 发现在num_workers设置num_workers参数，并没有提高速度，反而还降低了速度
# def safe_num_workers(dataset_len, max_workers=8):
#     return 0 if dataset_len <= 10000 else max_workers


def run(df, output_file) -> pd.DataFrame:
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
    return df


def run_big_file(file_name, output_file, split_file=True, chunksize=int(5e5)):
    """
    为运行大文件设计，把大文件分块运行。
    应对大文件长时间运行的挑战：大文件需要运行很长的时间，且不可中断程序。
    split_file: 是否分块处理文件; False代表不分块处理文件
    chunksize: 分块处理的行数。只有在split_file=True时生效。
    """
    basename = os.path.basename(file_name)
    file_size_mb = os.path.getsize(file_name) / (1024 * 1024)  # 转为MB

    if not split_file or file_size_mb < 1024:
        # 过小的文件，不需要分块处理，超过10GB是大文件
        df = pd.read_csv(file_name, low_memory=False)
        run(df, output_file)
        return
    
    print(f"loading {basename}")
    df_blocks = pd.read_csv(file_name, chunksize=chunksize, low_memory=False)
    cache_fold = "cache"
    output_parent_dir = os.path.dirname(output_file)
    os.makedirs(os.path.join(output_parent_dir, cache_fold), exist_ok=True)
    data = []
    for idx, tmp_df in enumerate(df_blocks):
        print(f"processing part {idx} of {basename}")
        tmp_output_file = os.path.join(
            output_parent_dir, cache_fold, f"{basename}_part_{idx}.csv"
        )
        if os.path.exists(tmp_output_file):
            data.append(pd.read_csv(tmp_output_file, low_memory=False))
        else:
            data.append(run(tmp_df, tmp_output_file))
    df = pd.concat(data)
    df.to_csv(output_file, index=False)

    # 最终的文本合并且导出完成，删除分块文件
    for tmp_output_file in glob.glob(os.path.join(output_parent_dir, cache_fold, "*")):
        os.remove(tmp_output_file)


if __name__ == "__main__":
    raw_data_dir = r"C:\Users\1\Desktop\战新企业\战新企业5"
    output_dir = r"C:\Users\1\Desktop\战新企业\outcome5"

    for sub_fold in os.listdir(raw_data_dir):
        p = os.path.join(raw_data_dir, sub_fold)
        for name in os.listdir(p):
            file_name = os.path.join(p, name)
            output_sub_dir = os.path.join(output_dir, sub_fold)
            os.makedirs(output_sub_dir, exist_ok=True)
            output_file = os.path.join(output_sub_dir, name)

            if not os.path.exists(output_file):
                # df = pd.read_csv(file_name, low_memory=False)
                # run(df, output_file)
                run_big_file(file_name, output_file)
            print(output_file, "done")
