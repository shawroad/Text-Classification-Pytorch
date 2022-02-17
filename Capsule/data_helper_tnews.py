"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-14
"""
import json
import pandas as pd
import torch
from build_vocab_tnews import clean_text
from torch.utils.data import Dataset


def load_data(path, label2id):
    text_list, label_list = [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            # 'label_desc': 'news_edu', 'sentence': '上课时学生手机响个不停，老师一怒之下把手机摔了，家长拿发票让老师赔，大家怎么看待这种事？'
            label = label2id[line['label_desc']]
            text = line['sentence']
            text_list.append(text)
            label_list.append(label)
    df = pd.DataFrame({'label': label_list, 'text': text_list})
    df['text'] = df['text'].astype(str)
    df.dropna(subset=['label', 'text'], inplace=True)
    # df.drop_duplicates(subset='ad', keep='first', inplace=True)  # 去重
    df.reset_index(drop=True, inplace=True)   # 重置索引
    df.loc[:, 'text'] = df['text'].map(clean_text)   # 清洗文本
    return df


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, vocab2id):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.label = dataframe.label
        self.vocab2id = vocab2id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        words_line = []
        text = self.text[item]
        token = self.tokenizer.tokenize(text)
        seq_len = len(token)
        for word in token:
            words_line.append(self.vocab2id.get(word, self.vocab2id.get('<UNK>')))
        return {'input_ids': words_line, 'label': self.label[item], 'seq_len': seq_len}


def pad_to_max_len(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    max_len = max([len(d['input_ids']) for d in batch])

    input_ids, labels, seq_len = [], [], []
    for item in batch:
        input_ids.append(pad_to_max_len(item['input_ids'], max_len=max_len))
        labels.append(item['label'])
        seq_len.append(item['seq_len'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.long)
    all_seq_len = torch.tensor(seq_len, dtype=torch.long)
    return all_input_ids, all_label_ids, all_seq_len

