"""
@file   : build_vocab.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-14
"""
import json
import re
import pickle as pkl
import numpy as np
from transformers.models.bert.tokenization_bert import BasicTokenizer


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


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    max_size = max_size - 2   # 取出pad和unk
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            text = line['sentence']
            text = clean_text(text)
            for word in tokenizer.tokenize(text):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx+2 for idx, word_count in enumerate(vocab_list)}
        vocab_dic['<PAD>'] = 0
        vocab_dic['<UNK>'] = 1
    return vocab_dic


if __name__ == "__main__":
    MAX_VOCAB_SIZE = 10000   # 词表长度限制
    label2id = json.load(open('../data/tnews_public/label2id.json'))
    train_data_path = '../data/tnews_public/train.json'
    # 1. 建立词表
    print('建立词表...')
    tokenizer = BasicTokenizer()
    word_to_id = build_vocab(train_data_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(word_to_id, open('./data/tnew_vocab.pkl', 'wb'))

    # 2. 加载词向量
    print('加载词向量...')
    emb_dim = 300   # 取决于加载预训练词向量维度大小
    pretrain_vocab_vec = '../vocab_vec/sgns.sogou.char'
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    # print(len(word_to_id))

    with open(pretrain_vocab_vec, 'r', encoding='utf8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip().split(' ')
            if line[0] in word_to_id:
                idx = word_to_id[line[0]]
                emb = [float(value) for value in line[1:]]
                embeddings[idx] = np.asarray(emb, dtype='float32')
    np.savez_compressed('./data/my.sgns.sogou.char', embeddings=embeddings)