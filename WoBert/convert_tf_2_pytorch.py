"""
@file   : convert_tf_2_pytorch.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-28
"""
from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch

path = './wobert_pretrain/chinese_wobert_plus_L-12_H-768_A-12'
tf_checkpoint_path = path + "/bert_model.ckpt"
bert_config_file = path + "/bert_config.json"
pytorch_dump_path = "./wobert_pretrain/pytorch_model.bin"

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file,
                                 pytorch_dump_path)
