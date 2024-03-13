import json  


def load_jsonl(file_path):  
    with open(file_path, 'r', encoding='utf-8') as file:  
        for line in file:  
            yield json.loads(line)

        
def doccano2BIO(file_name, output_file='out.txt'):
    def _get_pair():
        data = load_jsonl(file_name)
        for line in data:
            text = line['text']
            labels = ['O'] * len(text)
            for ent in line['entities']:
                label, start_offset, end_offset = ent['label'], ent['start_offset'], ent['end_offset']
                labels[start_offset] = 'B-' + label
                labels[start_offset+1: end_offset] = ['I-' + label] * (end_offset - start_offset - 1)
            yield text, labels
            
    with open(output_file, 'w+') as f:
        content = []
        for text, labels in _get_pair():
            s = []
            item = zip(list(text), labels)
            for line in item:
                s.append(' '.join(line) + '\n')
            s = ''.join(s)[:-1]
            content.append(s)
        content = "\n\n".join(content)
        f.write(content)


if __name__ == '__main__':
    doccano2BIO('guihua.jsonl', 'out.txt')

        