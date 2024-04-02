import json
from dataclasses import dataclass
from typing import List

@dataclass
class Ent:
    entity: str
    entity_type: str


@dataclass
class Rel:
    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str

    @classmethod
    def from_defaults(cls, head_ent: Ent, rel: str, tail_ent: Ent):
        return cls(
            head_ent.entity,
            head_ent.entity_type,
            rel,
            tail_ent.entity,
            tail_ent.entity_type
        )


@dataclass
class DataFormat:
    text: str
    entity: List[Ent]
    relation: List[Rel]

    def to_string(self):
        return json.dumps(
            {
                'text': self.text,
                'entity': str([ent.__dict__ for ent in self.entity]),
                'relation': str([rel.__dict__ for rel in self.relation])
            },
            ensure_ascii=False
        )


def doccano_trans(item):
    text = item['text']
    ents = item['entities']
    rels = item['relations']
    ent_d = {}

    for ent in ents:
        start_offset, end_offset = ent['start_offset'], ent['end_offset']
        name = text[start_offset:end_offset]
        label = ent['label']
        ent_d[ent['id']] = Ent(name, label)

    res_rel = []
    for rel in rels:
        from_id, to_id, rel_name = rel['from_id'], rel['to_id'], rel['type']
        head_ent, tail_ent = ent_d[from_id], ent_d[to_id]
        res_rel.append(
            Rel.from_defaults(head_ent, rel_name, tail_ent)
        )

    data_format = DataFormat(text, list(ent_d.values()), res_rel)
    return data_format.to_string()


if __name__ == '__main__':
    data = {'id': 6400, 'text': '扎实推进垃圾分类示范区创建，实现覆盖率100%，开展垃圾分类示范片区创建的街道占比达到100%',
            'Comments': [], 'entities': [{'id': 1662, 'label': '关键数据', 'start_offset': 26, 'end_offset': 39},
                                         {'id': 1663, 'label': '数值', 'start_offset': 43, 'end_offset': 47}],
            'relations': [{'id': 273, 'from_id': 1662, 'to_id': 1663, 'type': '达到'}]}

    print(doccano_trans(data))
