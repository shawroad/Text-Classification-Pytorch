"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-14
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class GRULayer(nn.Module):
    def __init__(self, hidden_size):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_size=300, hidden_size=hidden_size, bidirectional=True)

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for k in ih:
            nn.init.xavier_uniform_(k)
        for k in hh:
            nn.init.orthogonal_(k)
        for k in b:
            nn.init.constant_(k, 0)

    def forward(self, x):
        return self.gru(x)


class CapsuleLayer(nn.Module):
    def __init__(self, input_dim_capsule, num_capsule, dim_capsule, routings, activation='default'):
        super(CapsuleLayer, self).__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.t_epsilon = 1e-7    # 计算squash需要的参数
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = nn.ReLU(inplace=True)

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, input_dim_capsule, num_capsule * dim_capsule)))

    def forward(self, x):
        u_hat_vecs = torch.matmul(x, self.W)

        batch_size = x.size(0)
        input_num_capsule = x.size(1)
        u_hat_vecs = u_hat_vecs.view((batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])

        outputs = 0
        for i in range(self.routings):
            b = b.permute(0, 2, 1)
            c = F.softmax(b, dim=2)
            c = c.permute(0, 2, 1)
            b = b.permute(0, 2, 1)
            outputs = self.activation(torch.einsum('bij,bijk->bik', (c, u_hat_vecs)))
            if i < self.routings - 1:
                b = torch.einsum('bik,bijk->bij', (outputs, u_hat_vecs))
        return outputs

    def squash(self, x, axis=-1):
        s_squared_norm = (x ** 2).sum(axis, keepdim=True)
        scale = torch.sqrt(s_squared_norm + self.t_epsilon)
        return x / scale


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # 一些参数
        self.hidden_size = 128
        self.num_capsule = 10
        self.dim_capsule = 16
        self.routings = 5
        self.dropout_p = 0.25

        # 1. 词嵌入
        vocab_vec = torch.tensor(np.load('./data/my.sgns.sogou.char.npz')['embeddings'], dtype=torch.float)
        self.embedding = nn.Embedding.from_pretrained(vocab_vec, freeze=False)
        # self.embedding = nn.Embedding(args.n_vocab, args.embed, padding_idx=args.n_vocab - 1)

        # 2. 一层gru
        self.gru = GRULayer(hidden_size=self.hidden_size)
        self.gru.init_weights()  # 对gru的参数进行显性初始化

        # 3. capsule
        self.capsule = CapsuleLayer(input_dim_capsule=self.hidden_size * 2, num_capsule=self.num_capsule,
                                    dim_capsule=self.dim_capsule, routings=self.routings)

        # 4. 分类层
        self.classify = nn.Sequential(
            nn.Dropout(p=self.dropout_p, inplace=True),
            nn.Linear(self.num_capsule * self.dim_capsule, num_classes),
        )

    def forward(self, input_ids):
        batch_size = input_ids.size(0)

        embed = self.embedding(input_ids)
        # print(embed.size())  # torch.Size([2, 128, 300])

        output, _ = self.gru(embed)  # output.size()  torch.Size([2, 128, 256])

        cap_out = self.capsule(output)
        # print(cap_out.size())    # torch.Size([2, 10, 16])
        cap_out = cap_out.view(batch_size, -1)
        return self.classify(cap_out)

