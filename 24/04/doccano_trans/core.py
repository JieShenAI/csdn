import json

from utils import doccano_trans

file = 'data/test.jsonl'

data = []

with open(file, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(
            doccano_trans(json.loads(line))
        )

with open('out.jsonl', 'w', encoding='utf-8') as f:
    f.write('\n'.join(data))
