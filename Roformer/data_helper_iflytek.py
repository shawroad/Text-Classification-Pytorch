"""
@file   : data_helper_iflytek.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-22
"""
import re
import torch
import json
import pandas as pd
from torch.utils.data import Dataset


def clean_text(text):
    rule_url = re.compile(
        '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    )
    rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5]')
    rule_space = re.compile('\\s+')
    text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
    text = rule_url.sub(' ', text)
    text = rule_legal.sub(' ', text)
    text = rule_space.sub(' ', text)
    return text.strip()


def load_data(path, label2id):
    text_list, label_list = [], []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            # {'label': '11', 'label_des': '薅羊毛', 'sentence': '活动新用户领5元现金即日起成功'}
            label = label2id[line['label_des']]
            text = line['sentence']
            text_list.append(text)
            label_list.append(label)
    df = pd.DataFrame({'label': label_list, 'text': text_list})
    df['text'] = df['text'].astype(str)
    df.dropna(subset=['label', 'text'], inplace=True)
    # df.drop_duplicates(subset='ad', keep='first', inplace=True)  # 去重
    df.reset_index(drop=True, inplace=True)  # 重置索引
    df.loc[:, 'text'] = df['text'].map(clean_text)  # 清洗文本
    return df


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.text
        self.label = dataframe.label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            text=self.text[index],
            text_pair=None,
            add_special_tokens=True,
            return_token_type_ids=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.label[index]
        }


def pad_to_maxlen(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids


def collate_fn(batch):
    # 按batch进行padding获取当前batch中最大长度
    max_len = max([len(d['input_ids']) for d in batch])

    if max_len > 512:
        max_len = 512

    # 定一个全局的max_len
    # max_len = 128

    input_ids, attention_mask, token_type_ids, labels = [], [], [], []

    for item in batch:
        input_ids.append(pad_to_maxlen(item['input_ids'], max_len=max_len))
        attention_mask.append(pad_to_maxlen(item['attention_mask'], max_len=max_len))
        token_type_ids.append(pad_to_maxlen(item['token_type_ids'], max_len=max_len))
        labels.append(item['label'])

    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(attention_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(token_type_ids, dtype=torch.long)
    all_label_ids = torch.tensor(labels, dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids, all_label_ids


if __name__ == '__main__':
    label2id = json.load(open('../data/iflytek_public/label2id.json', 'r', encoding='utf8'))
    train_data_path = '../data/iflytek_public/train.json'
    train_df = load_data(train_data_path, label2id)
    print(train_df.shape)
    print(train_df.head())

    dev_data_path = '../data/iflytek_public/dev.json'
    dev_df = load_data(dev_data_path, label2id)
    print(dev_df.shape)
    print(dev_df.head())