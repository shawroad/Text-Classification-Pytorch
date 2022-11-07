"""
@file   : build_vocab.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-04
"""
import json
import re
import pickle as pkl
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BasicTokenizer


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


def build_vocab(all_text, tokenizer, max_size, min_freq):
    vocab_dic = {}
    max_size = max_size - 2  # 取出pad和unk

    for line in tqdm(all_text):
        for word in tokenizer.lcut(line):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx + 2 for idx, word_count in enumerate(vocab_list)}
    vocab_dic['<PAD>'] = 0
    vocab_dic['<UNK>'] = 1
    return vocab_dic


if __name__ == "__main__":
    MAX_VOCAB_SIZE = 10000   # 词表长度限制
    train_df = pd.read_csv('../data/train.csv', sep='\t')
    test_df = pd.read_csv('../data/test.csv', sep='\t')
    train_df['text'] = train_df['text'].map(clean_text)
    test_df['text'] = test_df['text'].map(clean_text)

    all_text = train_df['text'].tolist() + test_df['text'].tolist()

    # 1. 建立词表
    print('建立词表...')
    # tokenizer = BasicTokenizer()
    tokenizer = jieba
    word_to_id = build_vocab(all_text, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(word_to_id, open('./data/vocab.pkl', 'wb'))

    # 2. 加载词向量
    print('加载词向量...')
    emb_dim = 300   # 取决于加载预训练词向量维度大小
    pretrain_vocab_vec = '../weibo_word2vec/sgns.weibo.char'
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    print('总共有{}个词'.format(len(word_to_id)))
    count = 0
    with open(pretrain_vocab_vec, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
        for line in tqdm(lines):
            line = line.strip().split(' ')
            if line[0] in word_to_id:
                count += 1
                idx = word_to_id[line[0]]
                emb = [float(value) for value in line[1:]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed('./data/my.sgns.weibo.char', embeddings=embeddings)
    print('总共给{}个找到了词向量'.format(count))