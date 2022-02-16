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
        embed_dim = vocab_vec.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab_vec, freeze=False)
        # self.embedding = nn.Embedding(args.n_vocab, args.embed, padding_idx=args.n_vocab - 1)
        num_filters = 256
        filter_sizes = [2, 3, 4]
        dropout_rate = 0.5
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)   # torch.Size([64, 256, 27]) 
        x, _ = torch.max(x, dim=2)
        # print(x.size())   # torch.Size([64, 256)])
        return x

    def forward(self, input_ids):
        embed = self.embedding(input_ids)   # batch_size, max_len, hidden_size
        # print(embed.size())    # torch.Size([2, 21, 300])
        embed = embed.unsqueeze(1)   # torch.Size([2, 1, 21, 300])
        output = torch.cat([self.conv_and_pool(embed, conv) for conv in self.convs], 1)
        output = self.dropout(output)
        output = self.fc(output)
        return output
