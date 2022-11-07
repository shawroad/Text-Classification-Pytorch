"""
@file   : run_train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-04
"""
import os
import json
import random
import jieba
import pandas as pd
import pickle as pkl
from torch import nn
import torch.optim
import numpy as np
from tqdm import tqdm
from model import Model
from sklearn import metrics

from config import set_args
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from data_helper import CustomDataset, Collator
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert.tokenization_bert import BasicTokenizer


def set_seed():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def evaluate():
    eval_targets = []
    eval_predict = []
    model.eval()
    for step, batch in tqdm(enumerate(val_dataloader)):
        input_ids, labels, seq_len = batch
        if torch.cuda.is_available():
            input_ids, labels, seq_len = input_ids.cuda(), labels.cuda(), seq_len.cuda()
        with torch.no_grad():
            logits = model(input_ids)
        eval_targets.extend(labels.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
    eval_accuracy = metrics.accuracy_score(eval_targets, eval_predict)
    return eval_accuracy


if __name__ == '__main__':
    args = set_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = jieba
    # 加载数据集
    all_df = pd.read_csv(args.train_data_path, sep='\t')
    all_df['label'] = all_df['label'].map(lambda x: int(x))
    print(all_df.head())
    print('训练集的大小:', all_df.shape)

    # 训练数据集准备
    vocab2id = pkl.load(open('./data/vocab.pkl', 'rb'))

    skf = StratifiedKFold(n_splits=5)
    cv = 0
    texts = all_df['text']
    labels = all_df['label']
    collate_train_fn = Collator(is_train=True)
    collate_test_fn = Collator(is_train=False)
    for train_index, valid_index in skf.split(X=texts, y=labels):
        cv += 1  # 计算是第几折交叉
        train_df = all_df.loc[train_index.tolist(), :].reset_index(drop=True)
        val_df = all_df.loc[valid_index.tolist(), :].reset_index(drop=True)

        train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer, vocab2id=vocab2id)
        train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                      collate_fn=collate_train_fn)

        val_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer, vocab2id=vocab2id)
        val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=args.train_batch_size,
                                    collate_fn=collate_train_fn)

        model = Model()
        loss_func = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model.cuda()
            loss_func = loss_func.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                    num_training_steps=total_steps)
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            train_label, train_predict = [], []
            for step, batch in enumerate(train_dataloader):
                input_ids, labels, seq_len = batch
                if torch.cuda.is_available():
                    input_ids, labels, seq_len = input_ids.cuda(), labels.cuda(), seq_len.cuda()
                logits = model(input_ids, seq_len)
                loss = loss_func(logits, labels)
                loss.backward()
                print("当前轮次:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
                epoch_loss += loss

                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_label.extend(labels.cpu().detach().numpy().tolist())
                train_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())

            train_accuracy = metrics.accuracy_score(train_label, train_predict)
            eval_accuracy = evaluate()
            s = 'CV:{} | Epoch: {} | Loss: {:10f} | Train acc: {:10f} | Val acc: {:10f} '
            ss = s.format(cv, epoch, epoch_loss / len(train_dataloader), train_accuracy, eval_accuracy)

            logs_path = os.path.join(args.output_dir, 'logs.txt')
            with open(logs_path, 'a+') as f:
                ss += '\n'
                f.write(ss)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "base_model_cv_{}_epoch_{}.bin".format(cv, epoch))
            torch.save(model_to_save.state_dict(), output_model_file)
