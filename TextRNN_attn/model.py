"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-11-04
"""
import torch
import numpy as np
from torch import nn
from sys import platform
import torch.nn.functional as F


class CustomRNN(nn.Module):
    def __init__(self, input_size=300, dropout=0.2, bidirectional=True):
        super(CustomRNN, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=128, num_layers=2, bias=True,
                               batch_first=True, bidirectional=self.bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths.cpu(), batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if platform == "linux" or platform == "linux2":
            # for linux
            reordered_outputs = outputs.index_select(0, restoration_idx)
        else:
            # for win10
            reordered_outputs = outputs.index_select(0, restoration_idx.long())
        return reordered_outputs


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    sorted_seq_lens, sorting_index = sequences_lengths.sort(0, descending=descending)
    sorted_batch = batch.index_select(0, sorting_index)
    idx_range = torch.arange(0, len(sequences_lengths)).type_as(sequences_lengths)
    _, revese_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, revese_mapping)
    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        vocab_vec = torch.tensor(np.load('./data/my.sgns.weibo.char.npz')['embeddings'], dtype=torch.float)
        embed_dim = vocab_vec.size(1)
        self.embedding = nn.Embedding.from_pretrained(vocab_vec, freeze=False)
        # 考虑mask的rnn
        self.customlstm = CustomRNN()

        # 直接塞入rnn
        # hidden_size = 256
        # num_layers = 2
        # dropout = 0.5
        # self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        self.tanh = nn.Tanh()
        self.attn_alpha = nn.Linear(256, 1)

        self.classification = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, seq_len):
        emb = self.embedding(input_ids)
        # seq_output, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]
        # print(seq_output.size())   # torch.Size([64, 112, 512])
        # print(len(_))
        # print(_[0].size())   # torch.Size([4, 64, 256])  h
        # print(_[1].size())   # torch.Size([4, 64, 256])  c

        seq_output = self.customlstm(emb, seq_len)
        # print(seq_output.size())   # torch.Size([64, 121, 256])

        alpha = torch.softmax(self.attn_alpha(seq_output).squeeze(-1), dim=-1).unsqueeze(-1)
        # print(alpha.size())   # torch.Size([64, 121, 1])

        output = seq_output * alpha
        output = torch.sum(output, dim=1)
        output = torch.relu(output)

        logits = self.classification(output)
        # print(logits.size())   # torch.Size([64, 2])
        return logits
