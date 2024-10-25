import os

import pandas as pd
from pandas import DataFrame, read_csv

keys = ['t', 'i', 'j']
attrs = ['v']


def init_df(filename):
    data = read_csv(filename)
    delete_col = set(data.columns.values) - set(keys) - set(attrs)
    data = data.drop(list(delete_col), axis=1)

    d = {key: [] for key in keys + attrs}
    return data, pd.DataFrame.from_dict(d)


def is_equal(a: DataFrame, b: DataFrame):
    for key in keys:
        if a[key] != b[key]:
            return False
    return True


def add(a: DataFrame, b: DataFrame):
    assert is_equal(a, b)
    for attr in attrs:
        a[attr] += b[attr]


def insert(df: DataFrame, other: DataFrame):
    df.loc[len(df)] = other


def main(df, res):
    insert(res, df.iloc[0])
    for idx in range(1, len(df)):
        item = df.iloc[idx]
        if is_equal(item, res.iloc[-1]):
            add(res.iloc[-1], item)
        else:
            insert(res, item)


if __name__ == '__main__':
    source_folder = 'data/'
    output_folder = './output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            filename = os.path.join(root, file)
            print(filename, "处理中...")
            data, res_df = init_df(filename)
            main(data, res_df)
            res_df.to_csv(
                f := os.path.join(
                    output_folder,
                    os.path.basename(filename)),
            )
            print(res_df)
            print("转换完成---> ", f)