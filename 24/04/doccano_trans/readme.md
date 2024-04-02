
##  简介

Doccano 标注导出格式如下所示：
```python
{'id': 6400, 'text': '扎实推进垃圾分类示范区创建，实现覆盖率100%，开展垃圾分类示范片区创建的街道占比达到100%', 'Comments': [], 'entities': [{'id': 1662, 'label': '关键数据', 'start_offset': 26, 'end_offset': 39}, {'id': 1663, 'label': '数值', 'start_offset': 43, 'end_offset': 47}], 'relations': [{'id': 273, 'from_id': 1662, 'to_id': 1663, 'type': '达到'}]}
```
Doccano 标注导出格式的数据，不方便使用，无论是做信息抽取训练还是导入到图数据库中等，均无法直接使用；

故本文将其转为 [DeepKE](https://github.com/zjunlp/DeepKE/blob/main/example/llm/InstructKGC/data/NER/sample.json) 大模型训练数据格式，从而实现方便用户使用的目的。

虽然读者不一定使用DeepKE 训练大模型做信息抽取，但是转换后的数据格式，也能简化读者的数据转换工作。

<u>本文将Doccano标注导出的格式，转化为下述格式</u>

1. 如下是命名实体识别的数据格式：
	```python
	{"text": "相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。", "entity": [{"entity": "广州松日队", "entity_type": "组织机构"}, {"entity": "青岛海牛队", "entity_type": "组织机构"}]}
	```
2. 如下是SPO的数据格式：
   ```python
   {"text": "（电影里让我直接想起来的还有——吴宇森导演的《辣手神探》的殓房的秘门啊", "relation": [{"head": "辣手神探", "head_type": "影视作品", "relation": "导演", "tail": "吴宇森", "tail_type": "人物"}]}
   ```
   

csdn链接：[]()
