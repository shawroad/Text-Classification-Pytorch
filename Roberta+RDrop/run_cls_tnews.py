"""
@file   : run_cls.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-01-05
"""
import os
import copy
import torch
import json
import random
from tqdm import tqdm
import numpy as np
from torch import nn
from config import set_args
from model import Model
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.models.bert import BertTokenizer
from data_helper_tnews import load_data, CustomDataset, collate_fn
from transformers import AdamW, get_linear_schedule_with_warmup


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
        input_ids, input_mask, segment_ids, label_ids = batch
        if torch.cuda.is_available():
            input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
            label_ids = label_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)

        eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())

    eval_accuracy = metrics.accuracy_score(eval_targets, eval_predict)
    return eval_accuracy


def calc_loss(logits1, logits2, label_ids):
    # 分类损失
    loss_func = nn.CrossEntropyLoss()
    ce_loss = loss_func(logits1, label_ids) + loss_func(logits2, label_ids)
    
    # KL散度损失
    loss1 = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
    loss2 = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='none')
    alpha = 4
    kl_loss = (loss1 + loss2).mean() / 4 * alpha
    return ce_loss, kl_loss


if __name__ == '__main__':
    args = set_args()
    set_seed()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)

    # 加载数据集
    label2id = json.load(open('../data/tnews_public/label2id.json', 'r', encoding='utf8'))
    train_data_path = '../data/tnews_public/train.json'
    train_df = load_data(train_data_path, label2id)

    dev_data_path = '../data/tnews_public/dev.json'
    val_df = load_data(dev_data_path, label2id)
    
    print('训练集的大小:', train_df.shape)
    print('验证集的大小:', val_df.shape)

    # 训练数据集准备
    train_dataset = CustomDataset(dataframe=train_df, tokenizer=tokenizer)
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_fn)

    # 验证集准备
    val_dataset = CustomDataset(dataframe=val_df, tokenizer=tokenizer)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=True, batch_size=args.val_batch_size, collate_fn=collate_fn)

    total_steps = len(train_dataloader) * args.num_train_epochs

    num_train_optimization_steps = int(
        len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    model = Model(label_num=15)

    if torch.cuda.is_available():
        model.cuda()

    optimizer_grouped_parameters = [
        {"params": model.roberta.parameters()},
        # {'params': model.highway.parameters(), 'lr': args.learning_rate * 10},
        {"params": model.classifier.parameters(), 'lr': args.learning_rate * 10}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * total_steps,
                                                num_training_steps=total_steps)

    # loss_func = nn.BCEWithLogitsLoss()   # 普通的BCE损失

    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Batch size = %d" % args.train_batch_size)
    print("  Num steps = %d" % num_train_optimization_steps)
    for epoch in range(args.num_train_epochs):
        model.train()
        train_label, train_predict = [], []
        epoch_loss = 0

        # 制作tqdm对象
        for step, batch in enumerate(train_dataloader):
            # for step, batch in enumerate(train_dataloader):
            input_ids, input_mask, segment_ids, label_ids = batch
            if torch.cuda.is_available():
                input_ids, input_mask, segment_ids = input_ids.cuda(), input_mask.cuda(), segment_ids.cuda()
                label_ids = label_ids.cuda()

            logits1 = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
            logits2 = model(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
            ce_loss, kl_loss = calc_loss(logits1, logits2, label_ids)
            loss = kl_loss + ce_loss
            loss.backward()
            print("当前轮次:{}, 正在迭代:{}/{}, Loss:{:10f}".format(epoch, step, len(train_dataloader), loss))  # 在进度条前面定义一段文字
            # print('epoch:{}, step:{}, loss:{:10f}'.format(epoch, step, loss))
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
            epoch_loss += loss

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            logits = (logits1 + logits2) / 2

            train_label.extend(label_ids.cpu().detach().numpy().tolist())
            train_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())

        train_accuracy = metrics.accuracy_score(train_label, train_predict)
        eval_accuracy = evaluate()


        s = 'Epoch: {} | Loss: {:10f} | Train acc: {:10f} | Val acc: {:10f} '
        ss = s.format(epoch, epoch_loss / len(train_dataloader), train_accuracy, eval_accuracy)

        logs_path = os.path.join(args.output_dir, 'logs.txt')
        with open(logs_path, 'a+') as f:
            ss += '\n'
            f.write(ss)

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, "base_model_epoch_{}.bin".format(epoch))
        torch.save(model_to_save.state_dict(), output_model_file)
