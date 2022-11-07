"""
@file   : data_helper.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-04
"""
import json
import pandas as pd
import torch
import re
from torch.utils.data import Dataset


'''
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
'''


def clean_text(text):
    # 去除url
    rule_url = re.compile(
        '(https?://)?(www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    )
    # 去除杂乱的名字
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)   # 去重中间杂乱的名字
    rule_legal = re.compile('[^\\[\\]@#a-zA-Z0-9\u4e00-\u9fa5，！”？?~《》。#、；：“（）]')
    rule_space = re.compile('\\s+')
    text = str(text).replace('\\n', ' ').replace('\n', ' ').strip()
    text = rule_url.sub(' ', text)
    text = rule_legal.sub(' ', text)
    text = rule_space.sub(' ', text)
    return text.strip()


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, vocab2id, is_train=True):
        self.is_train = is_train
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.text
        if self.is_train:
            self.label = dataframe.label
        self.vocab2id = vocab2id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        words_line = []
        text = self.text[item]
        token = self.tokenizer.lcut(text)
        seq_len = len(token)
        for word in token:
            words_line.append(self.vocab2id.get(word, self.vocab2id.get('<UNK>')))

        if self.is_train:
            return {'input_ids': words_line, 'label': self.label[item], 'seq_len': seq_len}
        else:
            return {'input_ids': words_line, 'seq_len': seq_len}


class Collator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        max_len = max([len(d['input_ids']) for d in batch])
        input_ids, labels, seq_len = [], [], []
        for item in batch:
            input_ids.append(self.pad_to_max_len(item['input_ids'], max_len=max_len))
            seq_len.append(item['seq_len'])

            if self.is_train:
                labels.append(item['label'])

        all_input_ids = torch.tensor(input_ids, dtype=torch.long)
        all_seq_len = torch.tensor(seq_len, dtype=torch.long)

        if self.is_train:
            all_label_ids = torch.tensor(labels, dtype=torch.long)
            return all_input_ids, all_label_ids, all_seq_len
        else:
            return all_input_ids, all_seq_len

    def pad_to_max_len(self, input_ids, max_len, pad_value=0):
        if len(input_ids) >= max_len:
            input_ids = input_ids[:max_len]
        else:
            input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
        return input_ids
