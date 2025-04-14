from datasets import load_dataset
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm
from datasets import Dataset
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8000/v1/", model="gpt-3.5-turbo")
ds = load_dataset("fancyzhx/ag_news")

from settings import TEXT_CLS_NAMES, SAMPLE_NUM


class TextCLS(BaseModel):
    """
    Structured output for text classification
    """

    reason: str = Field(description="Classification rationale")
    label: str = Field(
        description='Text classification label (select exactly one): ["World", "Sports", "Business", "Science/Technology"]'
    )


structured_llm = llm.with_structured_output(TextCLS)


def func(item):
    label = item["label"]
    item["label_text"] = TEXT_CLS_NAMES[label]
    return item


test_dataset = ds["test"].map(func)
new_test_dataset = test_dataset.train_test_split(seed=42, test_size=SAMPLE_NUM)["test"]
prompt = PromptTemplate.from_template(
    "Classify the text into exactly one category with reasoning. Categories: World, Sports, Business, Science/Technology. \n{input}"
)

chain = prompt | structured_llm

## predict
gold_labels = []
pred_labels = []
reasons = []
queries = []
labels = []

for item in tqdm(new_test_dataset):
    text = item["text"]
    text_label = item["label_text"]
    ans: TextCLS = chain.invoke(text)
    if ans:
        gold_labels.append(text_label)
        pred_labels.append(ans.label)
        reasons.append(ans.reason)
        queries.append(text)
        labels.append(item["label"])

output_dataset = Dataset.from_dict(
    {
        "query": queries,
        "pred_label": pred_labels,
        "gold_label": gold_labels,
        "reason": reasons,
        "labels": labels,
    }
)

output_dataset.to_json("output/llm_struct.json")
