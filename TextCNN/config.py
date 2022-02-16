"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-14
"""
import argparse


def set_args():

    args = argparse.ArgumentParser('--textcnn')
    args.add_argument('--epochs', default=10, type=int, help='训练几轮')
    args.add_argument('--learning_rate', default=1e-3, type=int, help='学习率')
    args.add_argument('--train_batch_size', default=64, type=int, help='训练的批次大小')
    args.add_argument('--seed', default=43, type=int, help='随机种子大小')
    args.add_argument('--output_dir', default='./output', type=str, help='模型输出路径')
    args.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积聚')
    args.add_argument('--embedding_pretrained', default='./data/my.sgns.sogou.char.npz', type=str, help='词向量的路径')
    return args.parse_args()
