"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-04
"""
import argparse


def set_args():
    args = argparse.ArgumentParser('--textrnn_attn')
    args.add_argument('--epochs', default=5, type=int, help='训练几轮')
    args.add_argument('--learning_rate', default=1e-3, type=int, help='学习率')
    args.add_argument('--train_batch_size', default=64, type=int, help='训练的批次大小')
    args.add_argument('--seed', default=43, type=int, help='随机种子大小')
    args.add_argument('--output_dir', default='./output', type=str, help='模型输出路径')
    args.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚')
    args.add_argument('--embedding_pretrained', default='./data/my.sgns.weibo.char.npz', type=str, help='词向量的路径')
    args.add_argument('--train_data_path', default='../data/train.csv', type=str, help='训练集路径')
    args.add_argument('--test_data_path', default='../data/test.csv', type=str, help='测试集路径')
    return args.parse_args()