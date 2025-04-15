# 大模型结构化输出教程与评估

## 简介

结合langchain官方文档实现了一遍大模型的结构化输出。
相比官方文档，额外增加了prompt模板填充与few-shot提示。

### 开源

github 只有代码，没有数据。网盘链接包括数据示例。

* Github: [https://github.com/JieShenAI/csdn/tree/main/25/04/llm_with_structured_output](https://github.com/JieShenAI/csdn/tree/main/25/04/llm_with_structured_output)

  本文结构化输出教程地址：[https://github.com/JieShenAI/csdn/blob/main/25/04/llm_with_structured_output/struct_output.ipynb](https://github.com/JieShenAI/csdn/blob/main/25/04/llm_with_structured_output/struct_output.ipynb)

* 通过网盘分享的文件：llm_with_structured_output
  链接: [https://pan.baidu.com/s/1zb3TEXq7m7A2VS7xnT8zrg?free](https://pan.baidu.com/s/1zb3TEXq7m7A2VS7xnT8zrg?free)

## 大模型结构化输出

```python
from langchain_openai import ChatOpenAI

# 支持本地部署的大模型
# llm = ChatOpenAI(base_url="http://127.0.0.1:8000/v1/", model="gpt-3.5-turbo") 
llm_mini = ChatOpenAI(model="gpt-4o-mini")
llm_turbo = ChatOpenAI(model="gpt-3.5-turbo")
```

OPENAI_API_BASE 与 OPEN_API_KEY 参数已添加到系统环境变量中，故无需显式传参。



```python
from typing import List
from typing import Optional

from pydantic import BaseModel, Field


# Pydantic
class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(
        default=None, description="How funny the joke is, from 1 to 10"
    )
```

`Joke` 是大模型要结构化输出的对象。

```python
# 另外一种 Joke 的写法
from typing import Optional

from typing_extensions import Annotated, TypedDict


# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]

    # Alternatively, we could have specified setup as:

    # setup: str                    # no default, no description
    # setup: Annotated[str, ...]    # no default, no description
    # setup: Annotated[str, "foo"]  # default, no description

    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]
```

若 llm 是 `gpt-3.5-turbo`，可成功得到输出，若是`gpt-4o-mini` 则会报错。gpt-4o-mini 也具备function call与tool use的能力，但是能力逊色一些。

```python
llm_turbo.with_structured_output(Joke).invoke("Tell me a joke about cats")
```
输出：

```python
Joke(setup='Why was the cat sitting on the computer?', punchline='Because it wanted to keep an eye on the mouse!', rating=7)
```




```python
## 会报错
# llm_mini.with_structured_output(Joke).invoke("Tell me a joke about cats")
```



json 格式的 Joke:

```python
json_schema = {
    "title": "joke",
    "description": "Joke to tell user.",
    "type": "object",
    "properties": {
        "setup": {
            "type": "string",
            "description": "The setup of the joke",
        },
        "punchline": {
            "type": "string",
            "description": "The punchline to the joke",
        },
        "rating": {
            "type": "integer",
            "description": "How funny the joke is, from 1 to 10",
            "default": None,
        },
    },
    "required": ["setup", "punchline"],
}
```

```python
llm_turbo.with_structured_output(json_schema).invoke("Tell me a joke about cats")
```

输出：

```python
{'setup': 'Why was the cat sitting on the computer?',
 'punchline': 'Because it wanted to keep an eye on the mouse!'}
```

大模型按照上述json格式调用大模型，也能成功解析得到结构化对象的输出

### 绑定多个结构对象



```python
from typing import Union

class Man(BaseModel):
    """
    男人的信息
    """

    name: str = Field(description="姓名")
    age: str = Field(description="年龄")
    interest: str = Field(description="兴趣爱好")
    colthing: str = Field(description="上身衣服与下身衣服")
    height: str = Field(description="身高")


class Woman(BaseModel):
    """
    女人的信息
    """

    name: str = Field(description="姓名")
    age: str = Field(description="年龄")
    interest: str = Field(description="兴趣爱好")
    colthing: str = Field(description="上身衣服与下身衣服")
    height: str = Field(description="身高")



class Person(BaseModel):
    final_output: Union[Man, Woman]
```



```python
llm_turbo.with_structured_output(Person).invoke("帮我生成一个男人的信息")
```
输出：
```python
Person(final_output=Man(name='张伟', age='30', interest='运动，旅行，读书', colthing='白色衬衫，深色牛仔裤', height='175cm'))
```



```python
llm_turbo.with_structured_output(Person).invoke("帮我生成一个女人的信息")
```

输出：

```python
Person(final_output=Man(name='李华', age='28', interest='阅读，旅行，烹饪', colthing='白色衬衫和黑色裙子', height='165cm'))
```

`with_structured_output` 如果没有解析到对应的结构化对象，则返回None。

### Few-shot prompting

在前面的例子中，发现 `gpt-4o-mini`模型无法正确解析出结构化的对象，故使用少样本提示增强`gpt-4o-mini`模型的结构化输出能力。

**写法一**，直接写入到提示词中：

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

system_msg = """You are a hilarious comedian. Your specialty is knock-knock jokes. 
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?")."""

examples = """
example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}
""".strip()

prompt = PromptTemplate.from_template(
"""
{system_msg}

Here are some examples of jokes:
{examples}

example_user: {input}
""".strip()
)

# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])
```

由于使用了langchain的ChatPromptTemplate，故需要在提示词中使用`{{`对`{`进行转义。

langchain的提示词模板可以使用 `invoke`和`format` 进行提示词填充
查看填充完成后的提示词：

```python
prompt.invoke({
    "system_msg": system_msg,
    "examples": examples,
    "input": "what's something funny about woodpeckers",
}).messages
```



```python
print(
    prompt.format(
        system_msg=system_msg,
        examples=examples,
        input="what's something funny about woodpeckers",
    )
)
```

提示词输出，上述 invoke 与 format方法输出的提示词都是下述结果：

```python
You are a hilarious comedian. Your specialty is knock-knock jokes. 
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:
example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}
example_user: what's something funny about woodpeckers
```

调用大模型：

```python
few_shot_chain1 = prompt | llm_mini.with_structured_output(Joke)
few_shot_chain1.invoke(
    {
        "system_msg": system_msg,
        "examples": examples,
        "input": "what's something funny about woodpeckers",
    }
)
```

输出：

```python
Joke(setup='Woodpecker', punchline="Woodpecker knocking at your door? It's just trying to show you its new peck-formance!", rating=7)
```

`gpt-4o-mini` 模型经过少样本提示，就可以成功解析出结构化对象了。

****

**写法二**，`FewShotPromptTemplate`：

```python
import json
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

system_msg = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:""".strip()

# 定义格式化单个示例的 PromptTemplate
example_prompt = PromptTemplate(
    template="Q: {query}\nA: {{{answer}}}",
)

# 示例数据
examples = [
    {
        "query": "Tell me a joke about planes",
        "answer": {"setup": "Why don\'t planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2},
    },
    {
        "query": "Tell me another joke about planes",
        "answer": {"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10},
    }
]

for example in examples:
    example["answer"] = json.dumps(example["answer"])

# 构建 FewShotPromptTemplate
few_shot_prompt2 = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix=system_msg,
    suffix="Q: {input}\nA:",
)
```

```python
print(few_shot_prompt2.invoke({"input": "Now about caterpillars"}).text)
```

提示词输出：

```python
You are a hilarious comedian. Your specialty is knock-knock jokes. Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

Q: Tell me a joke about planes
A: {'setup': "Why don't planes ever get tired?", 'punchline': 'Because they have rest wings!', 'rating': 2}

Q: Tell me another joke about planes
A: {'setup': 'Cargo', 'punchline': "Cargo 'vroom vroom', but planes go 'zoom zoom'!", 'rating': 10}

Q: Now about caterpillars
A:
```

值得注意的是 examples的answer是一个字典。为了不让langchain报错，针对 PromptTemplate 我是这些写的：

```
"Q: {query}\nA: {{{answer}}}"
```

使用了三个括号，把answer包住。大家记住`{{`是`{`的转义，然后再去理解三个括号就行了。

```python
few_shot_chain2 = few_shot_prompt2 | llm_mini.with_structured_output(Joke)
few_shot_chain2.invoke(
    {
        "input": "what's something funny about woodpeckers",
    }
)
```

输出：

```python
Joke(setup='Woodpecker', punchline="Woodpecker who's always knocking on wood for good luck!", rating=8)
```



## PydanticOutputParser

并不是所有模型都支持 `with_structured_output`，有一些模型的function call的能力差很多，那么便可以使用 `PydanticOutputParser` 解析出结构化对象（需要在提示词中指定返回格式）。



~~~python
from langchain.output_parsers import PydanticOutputParser

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
    生成一个男人的信息。以json格式返回，包含姓名、年龄、兴趣爱好、上身衣服与下身衣服、身高。格式如下:
    ```json
    {{
        "name": "张三",
        "age": "20",
        "interest": "打篮球",
        "colthing": "白色T恤与黑色裤子",
        "height": "180cm"
    }}
    ```
    """.strip()
)
parser = PydanticOutputParser(pydantic_object=Man)
chain = prompt_template | llm_mini | parser
man_info = chain.invoke({})
# 判断是否为空
if man_info:
    print(man_info)
else:
    print("没有返回结果")
~~~

输出:

```python
Man(name='李四', age='28', interest='旅游与摄影', colthing='蓝色衬衫与卡其色长裤', height='175cm')
```

## 文本分类评估例子

大模型通过`with_structured_output`的方式获取结构化输出，本文想评估一下，使用function call获取机构化输出的方式是否会造成大模型的输出效果下降。

大模型实现文本分类的例子：

```python
# 实现一个文本分类例子
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

class TextCLS(BaseModel):
    """
    文本分类的结构化输出
    """

    keyword: List[str] = Field(description="问题中出现的与分类相关的关键词")
    reason: str = Field(description="分类的原因")
    label: str = Field(description="文本分类label")
    
schema = ["经济", "民生", "产业", "绿色发展", "军事", "其他"]

system_msg = "请你完成文本分类任务，按照要求完成关键词提取，输出分类原因与最终的类别。文本的类别是：{schema}"

# 定义格式化单个示例的 PromptTemplate
example_prompt = PromptTemplate(
    template="Q: {query}\nA: {answer}",
)

# 示例数据
examples = [
    {
        "query": "武汉市今年GDP上涨2%",
        "answer": '{{"keyword": ["GDP"], "reason": "GDP与经济相关", "label": "经济"}}',
    },
    {
        "query": "氢能产业园区的相关配套措施完善，园区内有很多氢能领域龙头企业",
        "answer": """{{
                "keyword": ["氢能产业园区", "氢能领域龙头企业"],
                "reason": "问题中的氢能产业园区与氢能领域龙头企业都与产业相关",
                "label": "产业",
            }}""".strip(),
    },
]

# 构建 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    prefix=system_msg,
    suffix="Q: {input}\nA:",
)
```

一次性传递全部参数，每次都要传schema，过于繁琐：

```python
prompt = few_shot_prompt.invoke(
    {
        "input": "武汉市今年GDP上涨2%",
        "schema": schema,
    }
)
print(prompt.text)
```

**部分提示词**: 设置文本分类的label，以后就不用每一次都传递schema进去，只需要输入问题即可：

```python
partial_prompt = few_shot_prompt.partial(schema=schema)
partial_prompt.invoke({"input":"武汉市今年GDP上涨2%"}) # 输出提示词
```

```python
chanin = partial_prompt | llm_mini.with_structured_output(TextCLS)
chanin.invoke("北京市今年的生产总值提高了5个百分点")
```

输出：

```python
TextCLS(keyword=['生产总值'], reason='生产总值与经济相关，反映经济增长情况', label='经济')
```

## 文本分类评估项目

调用在线大模型的API速度太慢，选择调用本地大模型，速度会快一点。

llamafactory api 部署本地大模型，部署脚本：

`qwen2.5_7B.yaml`:

```python
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
template: qwen
infer_backend: vllm
vllm_enforce_eager: true
```

`llamafactory-cli api qwen2.5_7B.yaml`

三种方法：

* 大模型 + re 正则匹配 `llm_infer.py`
* 大模型结构化输出 with_structured_output `llm_structured_output_infer.py`
* vllm_infer 推理 `vllm_infer/`

### vllm_infer 推理 

大模型推理速度很快，但是步骤繁琐一些

* `vllm_infer/1.build_vllm_dataset.ipynb` 构建 alpaca 样式的数据集 ag_news_test，方便 llamafactory直接加载

* `vllm_infer/infer.sh` 大模型 vllm_infer 推理的脚本：

  ```python
  python vllm_infer.py \
              --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
              --template qwen \
              --dataset "ag_news_test" \
              --dataset_dir data \
              --save_name output/vllm_ag_news_test.json \
              > logs/vllm_ag_news_test.log 2>&1
  ```

  vllm_infer.py 来自 [https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/vllm_infer.py](https://github.com/hiyouga/LLaMA-Factory/blob/main/scripts/vllm_infer.py)

### 评估

`llm_result_eval.ipynb`: 对三种方法进行评估

|      |     method | precision |   recall |       f1 | support/数量 | processing_time/s |
| ---: | ---------: | --------: | -------: | -------: | -----------: | ----------------: |
|    1 |        llm |  0.841183 | 0.807035 | 0.794224 |          995 |              1600 |
|    2 | vllm_infer |  0.841379 | 0.809810 | 0.797484 |          999 |                41 |
|    3 | llm_struct |  0.838213 | 0.808809 | 0.803743 |          999 |              1339 |



* llm： 大模型api调用 + re 正则表达式匹配输出结果
* vllm_infer: vllm_infer 推理 + re 正则表达式匹配输出结果
* llm_struct: with_structured_output 结构化输出

从结果来看，三种方法做文本分类的precision、recall、f1的效果都差不多。

* support代表正确的结构化输出的数量，随机采样了1000条数据作为评估的数据集，三种方法结构化输出的能力都达到了99%。
* processing_time 代表推理速度，单位是秒。llm`与`llm_struct`的API调用，都采取的同步调用，故速度会慢很多，若采取异步速度会快三倍。当然 vllm_infer 的速度是最快的，只需要41秒，另外两种同步API调用速度都要将近半小时。

## 总结

采用了 function call 的 with_structured_output 的文本分类效果与其他方法都差不多。在提示词写准确后，with_structured_output  可以放心使用，能简化代码，不用自己再写正则表达式抽取大模型返回的结果。

从速度方面看，采用 vllm_infer 速度最快，而且文本分类效果也不会下降。

虽然 with_structured_output API调用速度很慢，但相比 vllm_infer 在 langgraph的agent、工作流编排方面，使用会方便很多。

