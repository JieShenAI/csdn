# LLamafactory 批量推理与异步 API 调用效率对比实测

## 背景

在阅读 LLamafactory 的文档时候，发现它支持批量推理:
 [推理.https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/inference.html](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/inference.html) 。

于是便想测试一下，它的批量推理速度有多快。本文实现了 下述两种的大模型推理，并对比了他们速度差别：
- LLamafactory API 部署，并通过 python 异步调用；
- LLamafactory 批量推理；


## 数据集构造
LLamafactory 批量推理的数据集，需要在 `data/dataset_info.json` 文件中完成注册。

`build_dataset.ipynb`:

```python
import json
import random
from typing import List


def generate_arithmetic_expression(num: int):
    # 定义操作符和数字范围，除法
    operators = ["+", "-", "*"]
    expression = (
        f"{random.randint(1, 100)} {random.choice(operators)} {random.randint(1, 100)}"
    )
    num -= 1
    for _ in range(num):
        expression = f"{expression} {random.choice(operators)} {random.randint(1, 100)}"
    result = eval(expression)
    expression = expression.replace("*", "x")
    return expression, result


def trans2llm_dataset(
    texts: List[str],
    labels: List[str],
    output_file,
    instruction="",
    prompt_template="",
    replace_kw="",
):

    data = []
    for text, label in zip(texts, labels):
        if replace_kw and prompt_template:
            text = prompt_template.replace(replace_kw, text)

        d = {
            "instruction": instruction,
            "input": text,
            "output": label,
        }
        data.append(d)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

prompt_template = """
    你是一名擅长数学运算的助手，负责逐步推理并解决四则运算问题。请按照以下步骤进行：

    1. 阅读并理解问题。
    2. 分步计算，逐步解决问题。
    3. 给出最终的结果。
    4. 按照 JSON 格式输出结果，包括：
    - reason: 详细的推理过程。
    - infer: 最终的计算结果。

    问题：{question}
    请给出分析和结果。
    """.strip()

texts = []
labels = []

for _ in range(100):
    text, label = generate_arithmetic_expression(2)
    texts.append(text)
    labels.append(label)

trans2llm_dataset(
    texts=texts,
    labels=labels,
    output_file="calculate.json",
    prompt_template=prompt_template,
    replace_kw="{question}",
)
```
上述程序运行后，得到了下图所示的数据集：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/56865d17a7aa46aea91801e93bafb619.png)

把该数据集在`dataset_info.json`中使用绝对路径注册：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bd074d6f7646467193eacaad75a0f30a.png)
## LLamafactory 批量推理

### yaml 参数设置
```python
# examples/train_lora/llama3_lora_predict.yaml
### model
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct
adapter_name_or_path: saves/llama3-8b/lora/sft

# deepspeed: examples/deepspeed/ds_z3_config.yaml # deepspeed配置文件

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
# eval_dataset: identity,alpaca_en_demo
eval_dataset: calculate
template: qwen
cutoff_len: 1024
# max_samples: 50
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: 模型预测结果的输出路径
overwrite_output_dir: true

### eval
per_device_eval_batch_size: 1
predict_with_generate: true
ddp_timeout: 180000000
```

参数介绍：
- eval_dataset: identity,alpaca_en_demo
- max_samples: 50

`eval_dataset` 是待预测/评估的数据集，支持填写多个数据集;
`max_samples` 代表从数据集中随机采样的数量；若不填，默认是全部数据集;


### 批量推理启动

由于要用到数据集，为了使得`LLaMA-Factory` 能够找到该数据集，故要在`LLaMA-Factory` 项目路径下运行命令，不然就会报'data/dataset_info.json 找不到的错误：
```
ValueError: Cannot open data/dataset_info.json due to [Errno 2] No such file or directory: 'data/dataset_info.json'.
```

cd 切换到 LLaMA-Factory 项目路径下，确保当前路径有 data 文件夹：
```
cd xxx/.../LLaMA-Factory
```

```shell
nohup llamafactory-cli train /绝对路径/csdn/24/11/llamafactory_batch_infer/batch_infer.yaml
```


但是 llamafactory 的批量推理不支持 vllm，所以推理速度有点慢，甚至还不如异步的API调用。

100%|██████████| 100/100 [04:42<00:00,  2.82s/it]


下述批量推理完，输出的结果：


使用批量推理的会输出一些文件：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1fc2636b252142a185af1b0417dfcc6b.png)
预测结果保存在 `predict_results.json`中：
```python
{"prompt": "system\nYou are a helpful assistant.\nuser\n你是一名擅长数学运算的助手，负责逐步推理并解决四则运算问题。请按照以下步骤进行：\n\n    1. 阅读并理解问题。\n    2. 分步计算，逐步解决问题。\n    3. 给出最终的结果。\n    4. 按照 JSON 格式输出结果，包括：\n    - reason: 详细的推理过程。\n    - infer: 最终的计算结果。\n\n    问题：58 + 15 + 17\n    请给出分析和结果。\nassistant\n", "label": "90", "predict": "```json\n{\n    \"reason\": \"首先，我们需要解决加法问题58 + 15 + 17。我们可以按照从左到右的顺序逐步计算。首先，计算58 + 15，得到73。然后，将73与17相加，最终结果是90。\",\n    \"infer\": 90\n}\n```"}
{"prompt": "system\nYou are a helpful assistant.\nuser\n你是一名擅长数学运算的助手，负责逐步推理并解决四则运算问题。请按照以下步骤进行：\n\n    1. 阅读并理解问题。\n    2. 分步计算，逐步解决问题。\n    3. 给出最终的结果。\n    4. 按照 JSON 格式输出结果，包括：\n    - reason: 详细的推理过程。\n    - infer: 最终的计算结果。\n\n    问题：7 + 35 + 9\n    请给出分析和结果。\nassistant\n", "label": "51", "predict": "```json\n{\n  \"reason\": \"首先，我们从左到右依次进行加法运算。首先计算7加35，得到42。然后将42加上9，得到最终结果51。\",\n  \"infer\": 51\n}\n```"}
{"prompt": "system\nYou are a helpful assistant.\nuser\n你是一名擅长数学运算的助手，负责逐步推理并解决四则运算问题。请按照以下步骤进行：\n\n    1. 阅读并理解问题。\n    2. 分步计算，逐步解决问题。\n    3. 给出最终的结果。\n    4. 按照 JSON 格式输出结果，包括：\n    - reason: 详细的推理过程。\n    - infer: 最终的计算结果。\n\n    问题：92 - 50 + 48\n    请给出分析和结果。\nassistant\n", "label": "90", "predict": "```json\n{\n    \"reason\": \"首先，我们需要按照数学中的运算顺序规则来解决这个问题。这个问题中只包含加法和减法，没有乘除运算，所以我们按照从左到右的顺序进行计算。首先执行92减去50，得到42。然后将得到的结果42加上48，最终结果为90。\",\n    \"infer\": 90\n}\n```"}
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1534133ef4344fbb969e5824562709d9.png)

## LLamafactory API 部署

可以，参考我们的前一篇文章，关于 python 异步调用 API 的文章。
- [大模型 API 异步调用优化：高效并发与令牌池设计实践
.https://blog.csdn.net/sjxgghg/article/details/143858730](https://blog.csdn.net/sjxgghg/article/details/143858730)
- [LLamafactory API部署与使用异步方式 API 调用优化大模型推理效率
.https://blog.csdn.net/sjxgghg/article/details/144049692](https://blog.csdn.net/sjxgghg/article/details/144049692)

完成 大模型 API 的部署：
```shell
llamafactory-cli api vllm_api.yaml 
```

100%|██████████| 100/100 [00:14<00:00,  6.76it/s]


由于 llamafactory 的批量推理不支持 vllm ，所以导致速度很慢，100条数据推理完，总计用时4分吧42秒。

而使用 异步的 API 调用的方式，仅仅用时14秒，就完成了100条数据的推理。

## 结论

lamafactory 的批量推理不支持 vllm 速度很慢。还是建议 lamafactory 把大模型部署成 API 服务，使用异步的调用API更快一点。

当然最快的还是使用 vllm 批量推理，这样会麻烦一些。使用 vllm 针对大模型进行推理会有一些繁琐的配置。比如参考：[llama-factory SFT 系列教程 (四)，lora sft 微调后，使用vllm加速推理
.https://blog.csdn.net/sjxgghg/article/details/137993809](https://blog.csdn.net/sjxgghg/article/details/137993809)

我个人喜欢的流程是：
1. 使用 LLamafactory 微调模型；
2. LLamafactory vllm api 部署模型；
3. 使用异步调用 API。

## 项目开源

[https://github.com/JieShenAI/csdn/tree/main/24/11/llamafactory_batch_infer](https://github.com/JieShenAI/csdn/tree/main/24/11/llamafactory_batch_infer)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8a13798bf56e4f0a9d01433f9e142c89.png)
- vllm_api.yaml 是 llamafactory API部署，供API异步调用的配置
- build_dataset.ipynb 构建数据集
- async_infer.ipynb 异步调用调试代码，因为 .ipynb 运行异步有点麻烦 
- async_infer.py 异步调用的代码
