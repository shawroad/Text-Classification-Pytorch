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
        num_filters = 250
        
        self.conv_region = nn.Conv2d(1, num_filters, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(num_filters, num_filters, (3, 1), stride=1)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))   # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))   # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters,  num_classes)
        
    def _block(self, x):
        x = self.padding2(x)   # torch.Size([64, 250, 43, 1])
        px = self.max_pool(x)
        
        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)
        
        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        
        x = x + px
        return x

    def forward(self, input_ids):
        embed = self.embedding(input_ids)
        # print(embed.size())   # torch.Size([64, 44, 300])

        embed = embed.unsqueeze(1)   # batch_size, 
        # print(embed.size())   # torch.Size([64, 1, 44, 300])
        
        x = self.conv_region(embed)   # batch_size, num_filters, seq_len - 3 + 1, 1   => torch.Size([64, 250, 42, 1])
        # print(x.size())    # torch.Size([64, 250, 42, 1])
        
        x = self.padding1(x)   # seq_len - 3 + 1 又变成了seq_len   64,250,44,1
        x = self.relu(x)
        x = self.conv(x)
        # print(x.size())  # torch.Size([64, 250, 42, 1])

        x = self.padding1(x)
        x = self.relu(x)
        x = self.conv(x)
        # print(x.size())   # torch.Size([64, 250, 42, 1])
        while x.size()[2] > 1:
            x = self._block(x)
        x = x.squeeze()    # torch.Size([64, 250])  
        x = self.fc(x)   
        return x

