# 参考链接
# https://blog.csdn.net/li_qili_qi/article/details/103700789
# https://huggingface.co/docs/evaluate/base_evaluator

import json
from dataclasses import dataclass

ent_class = ["PER", "ORG", "LOC"]


@dataclass
class Node:
    # 默认值
    predict_right_num: int = 0
    predict_num: int = 0
    label_num: int = 0


# 添加额外标签
def add_extra_labels(input_file, output_file):
    def _add_extra_labels(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                label_data = eval(data['label'])

                extra_labels = {
                    ent: []
                    for ent in ent_class
                }

                for ent in label_data:
                    entity = ent['entity']
                    entity_type = ent['entity_type']
                    if entity_type in ent_class:
                        extra_labels[entity_type].append(entity)
                data['extra_label'] = extra_labels
                yield data

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in _add_extra_labels(input_file):
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def compute(input_file):
    # 精确率：识别出正确的实体数 / 识别出的实体数

    # 集合与
    with open(input_file, 'r', encoding='utf-8') as f:
        total_ent = {
            ent: Node()
            for ent in ent_class
        }
        error = 0
        for line in f:
            data = json.loads(line)
            extra_labels = data['extra_label']
            try:
                predict = eval(data['output'])
            except:
                error += 1
                continue

            for ent_name in ent_class:
                extra_s = set(extra_labels[ent_name])
                predict_s = set(predict[ent_name])
                total_ent[ent_name].predict_right_num += len(extra_s & predict_s)
                total_ent[ent_name].predict_num += len(predict_s)
                total_ent[ent_name].label_num += len(extra_s)

    for ent in ent_class:
        acc = total_ent[ent].predict_right_num / (total_ent[ent].predict_num + 1e-6)
        recall = total_ent[ent].predict_right_num / (total_ent[ent].label_num + 1e-6)
        f1 = 2 * acc * recall / (acc + recall)

        print(f'{ent} acc: {acc:.4f} recall: {recall:.4f} f1: {f1:.4f}')


if __name__ == '__main__':
    
    input_file = 'data/predict_data.json'
    output_file = 'data/data.json'
    add_extra_labels(input_file, output_file)
    compute(output_file)
