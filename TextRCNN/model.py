"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-14
"""
import numpy as np
import torch
from config import set_args
from torch import nn
import torch.nn.functional as F

args = set_args()


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        vocab_vec = torch.tensor(np.load('./data/my.sgns.sogou.char.npz')['embeddings'], dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(vocab_vec, freeze=False)
        # self.embedding = nn.Embedding(args.n_vocab, args.embed, padding_idx=args.n_vocab - 1)

        self.embed = 300
        self.hidden_size = 256
        self.num_layers = 2
        self.dropout = 0.6

        self.lstm = nn.LSTM(self.embed, self.hidden_size, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * 2 + self.embed, num_classes)

    def forward(self, input_ids):
        embed = self.embedding(input_ids)   # batch_size, max_len, hidden_size
        # print(embed.size())    # torch.Size([2, 21, 300])

        out, _ = self.lstm(embed)
        # print(out.size())    # torch.Size([2, 21, 512])

        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        # print(out.size())    # torch.Size([2, 21, 812])
        out, _ = torch.max(out, dim=1)    # max-pooling
        out = self.fc(out)
        return out