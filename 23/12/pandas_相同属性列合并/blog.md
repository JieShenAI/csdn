@[toc]
# 根据相同属性合并pandas行

以如下表格数据为例，针对`t, i, j`相同的行，对其后的`v`属性数据实现相加。

`data.csv`的数据如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/839e5646e35d4e2a82545fe05195773f.png)

```python
import os

import pandas as pd
from pandas import DataFrame, read_csv
filename = "文件路径/data.csv"
```

## 数据加载

若数据量大，可只加载几行数据
```python
head = read_csv(filename, nrows=10)
```

`init_df`方法的作用：读取文件数据，创建一个新的容器存放最终的数据

根据`keys = ['t', 'i', 'j']`，对key相同的属性行将其`attrs=['v']`的属性值进行相加。

只留下`keys`,`attrs`属性列，使用`data.drop`删除掉其他属性列。

```python
keys = ['t', 'i', 'j']
attrs = ['v']

def init_df(filename):
    data = read_csv(filename)
    delete_col = set(data.columns.values) - set(keys) - set(attrs)
    data = data.drop(list(delete_col), axis=1)

	# 创建一个空容器，用于存放最终新的pandas数据
    d = {key: [] for key in keys + attrs}
    return data, pd.DataFrame.from_dict(d)
```

`data_df`:  读取的文件数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/516a2a616cba4958801d44bd64c3689b.png)

`new_df`: 空的 pandas.DataFrame 容器

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4cea8bc9b9b94b84bc3a6cc680dfb62d.png)
csv的数据是按照`keys = ['t', 'i', 'j']`的顺序排序好的。

故遍历`data_df`的每一行数据，将其与`new_df`最后一行数据的`keys`进行比较，若其相等则把`v`对应的属性值相加。
故需要实现如下几个函数
* 判断两行数据的`keys`是否相等， `is_equal()`
* 插入一行新数据到`new_df`，`insert()`

## is_equal() 方法

```python
def is_equal(a: DataFrame, b: DataFrame) -> bool:
    for key in keys:
        if a[key] != b[key]:
            return False
    return True
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a8f193a6ea9c4ffca379099951c6f632.png)

```python
# 属性值相加函数
def add(a: DataFrame, b: DataFrame):
    assert is_equal(a, b)
    for attr in attrs:
        a[attr] += b[attr]

# 插入一行数据到容器最后一行
def insert(df: DataFrame, other: DataFrame):
    df.loc[len(df)] = other
```

## 主函数

```python
def main(df, res):
    insert(res, df.iloc[0])
    for idx in range(1, len(df)):
        item = df.iloc[idx]
        if is_equal(item, res.iloc[-1]):
            add(res.iloc[-1], item)
        else:
            insert(res, item)

main(data_df, new_df)
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5871c61d39f948ef8ee40b4ce5b228c8.png)

点击查看此 [ipynb]() 格式代码
## 完整代码如下
下图是一个完整的封装完成的py代码
处理`source_folder`文件夹下的所有表格数据，将其处理结果保存到`output`文件下。
```python
import os

import pandas as pd
from pandas import DataFrame, read_csv

keys = ['t', 'i', 'j']
attrs = ['k']


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
    source_folder = '/Users/jshen/Desktop/jie/data'
    output_folder = 'output'
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
            print("转换完成---> ", f)
```
