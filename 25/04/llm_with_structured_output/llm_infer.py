import re
import json
from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
from langchain_core.prompts import PromptTemplate
from langchain.schema import AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8000/v1/", model="gpt-3.5-turbo")
ds = load_dataset("fancyzhx/ag_news")

from settings import TEXT_CLS_NAMES, SAMPLE_NUM

prompt = PromptTemplate.from_template(
    """
    Classify the given text into exactly one category, providing clear reasoning for your choice. The available categories are: World, Sports, Business, Science/Technology.
    Text to classify:
    {input}
    Output the classification reasoning in reason and the selected category in label. Return the response in JSON format as follows:
    ```json
    {{
        "reason" : "Classification rationale",
        "label" : "Text classification label (select exactly one): ["World", "Sports", "Business", "Science/Technology"]"
    }}
    ```
    """
)

def func(item):
    label = item["label"]
    item["label_text"] = TEXT_CLS_NAMES[label]
    return item


test_dataset = ds["test"].map(func)
new_test_dataset = test_dataset.train_test_split(seed=42, test_size=SAMPLE_NUM)["test"]


chain = prompt | llm

## predict
gold_labels = []
pred_labels = []
reasons = []
queries = []
labels = []


# Custom parser
def extract_json(message: AIMessage):
    pattern = r"```json(.*?)```"
    matches = re.search(pattern, message.content, re.DOTALL)
    if matches:
        res = matches.group(1)
        try:
            d = json.loads(res.strip())
        except Exception as e:
            print(e.args)
            d = {}
        finally:
            return d
    else:
        return {}


for item in tqdm(new_test_dataset):
    text = item["text"]
    text_label = item["label_text"]
    ans = chain.invoke(text)
    ans = extract_json(ans)
    if ans:
        gold_labels.append(text_label)
        pred_labels.append(ans["label"])
        reasons.append(ans["reason"])
        queries.append(item["text"])
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

output_dataset.to_json("output/llm.json")
