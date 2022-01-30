import json


if __name__ == '__main__':
    with open('labels.json', 'r', encoding='utf8') as f:
        lines = f.readlines()
        # {"label": "100", "label_desc": "news_story"}
        label2id = {}
        for i, line in enumerate(lines):
            line = json.loads(line)
            label = line['label_desc']
            label2id[label] = i

    json.dump(label2id, open('label2id.json', 'w', encoding='utf8'))
