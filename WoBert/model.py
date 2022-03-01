"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-28
"""
import torch
from torch import nn
from pdb import set_trace
from transformers.models.bert import BertModel, BertConfig


class Classifier(nn.Module):
    # 加个全连接 进行分类
    def __init__(self, num_cls):
        super(Classifier, self).__init__()
        self.dense1 = torch.nn.Linear(768, 384)
        self.dense2 = torch.nn.Linear(384, num_cls)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class Model(nn.Module):
    def __init__(self, label_num):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained('./wobert_pretrain/config.json')
        self.config.output_hidden_states = True   # 输出所有的隐层
        self.config.output_attentions = True  # 输出所有注意力层计算结果
        self.roberta = BertModel.from_pretrained('./wobert_pretrain', config=self.config)

        num_cls = label_num
        # self.highway = Highway(size=768, num_layers=3)
        self.classifier = Classifier(num_cls)

    def forward(self, input_ids, attention_mask, segment_ids):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # output[0].size(): batch_size, max_len, hidden_size
        # output[1].size(): batch_size, hidden_size
        # len(output[2]): 13, 其中一个元素的尺寸: batch_size, max_len, hidden_size
        # len(output[3]): 12, 其中一个元素的尺寸: batch_size, 12层, max_len, max_len

        cls_output = output[1]
        # hw_output = self.highway(cls_output)
        logits = self.classifier(cls_output)
        return logits
