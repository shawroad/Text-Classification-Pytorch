import json


if __name__ == '__main__':
    label2id = {}
    with open('labels.json', 'r', encoding='utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            line = json.loads(line)
            label = line['label_des']
            label2id[label] = i
    json.dump(label2id, open('label2id.json', 'w', encoding='utf8'), ensure_ascii=False)



