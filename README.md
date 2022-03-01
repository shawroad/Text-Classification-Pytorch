- [中文文本分类模型集锦](#中文文本分类模型集锦)
  - [数据说明](#数据说明)
  - [实验结果:](#实验结果)
  - [模型说明](#模型说明)
    - [1. FastText](#1-fasttext)
      - [安装](#安装)
      - [训练](#训练)
    - [2. NEZHA](#2-nezha)
      - [配置](#配置)
    - [3. Roberta](#3-roberta)
      - [配置](#配置-1)
      - [训练](#训练-1)
    - [4. MengZi](#4-mengzi)
      - [配置](#配置-2)
      - [训练](#训练-2)
    - [5. TextRCNN](#5-textrcnn)
      - [配置](#配置-3)
      - [训练](#训练-3)
  
 
# 中文文本分类模型集锦
## 数据说明
数据来源: https://github.com/CLUEbenchmark/CLUE

1. tnews_public数据(短文本分类)

该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。  
数据量：训练集(53,360)，验证集(10,000)，测试集(10,000)

2. iflytek_public数据(长文本分类) 

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。  
数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)

## 实验结果
| 模型 | iflytek(各类应用主题数据 长文本) Accuracy | tnews(今日头条新闻数据集 短文本) Accuracy | Avg |
| :-: | :-: | :-: | :-: | 
| FastText | 53.09 | 51.03 | 52.06 | 
| TextCNN | 55.86 | 53.82 | 54.84 |
| DPCNN | 49.55 | 51.89 | 50.72 |
| TextRCNN| 52.25 | 51.82 | 52.03 |
| Capsule | 55.09 | 51.14 | 53.11 |
| MengZi-base | 60.44 | 57.40  | 58.92 | 
| Roberta-base | 60.75 | 57.27 | 59.01 | 
| Roberta+MultiDrop| 60.71 | 57.94 | 59.32 |
| Roberta+RDrop| **60.94** | 58.17 | 59.55 |
| Roberta+HighWay| 58.09 | 57.54 | 57.81 |
| Wobert | 60.86 | **58.44** | **59.65**|

## 模型说明

### 1. FastText
####  安装
pip install fasttext==0.9.2 -i https://pypi.douban.com/simple/

####  训练 
iflytek数据集  
1. 准备数据: python data_helper_iflytek.py
2. 训练模型: python run_train.py
3. 推理评测: python inference.py 

tnews数据集  
1. 准备数据: python data_helper_tnews.py
2. 训练模型: python run_train.py
3. 推理评测: python inference.py 


### 2. NEZHA
#### 配置
transformers==4.11.3  
tokenizers==0.10.1  
预训练模型下载地址: https://github.com/lonePatient/NeZha_Chinese_PyTorch

### 3. Roberta
#### 配置
transformers==4.11.3  

预训练模型下载地址: https://github.com/ymcui/Chinese-BERT-wwm

#### 训练
iflytek数据集  
CUDA_VISIBLE_DEVICES=0 python run_cls_iflytek.py   # 直接做了验证 

tnews数据集  
CUDA_VISIBLE_DEVICES=0 python run_cls_tnews.py   # 直接做了验证

### 4. MengZi
#### 配置
transformers==4.11.3  
预训练模型下载地址: https://github.com/Langboat/Mengzi

#### 训练
iflytek数据集  
CUDA_VISIBLE_DEVICES=0 python run_cls_iflytek.py   # 直接做了验证

tnew数据集  
CUDA_VISIBLE_DEVICES=0 python run_cls_tnews.py   # 直接做了验证


### 5. TextRCNN
#### 配置
词向量下载: https://github.com/Embedding/Chinese-Word-Vectors

#### 训练
iflytek数据集
1. python build_vocab_iflytek.py   # 构建词表和加载词向量
2. CUDA_VISIBLE_DEVICES=0 python run_train_iflytek.py    # 训练+验证

tnew数据集
1. python build_vocab_tnews.py   # 构建词表和加载词向量
2. CUDA_VISIBLE_DEVICES=0 python run_train_tnews.py   # 训练+验证
 
### 6. Wobert
#### 配置
预训练模型下载: https://github.com/ZhuiyiTechnology/WoBERT
#### 训练
iflytek数据集  
CUDA_VISIBLE_DEVICES=0 python run_cls_iflytek.py   # 直接做了验证

tnew数据集  
CUDA_VISIBLE_DEVICES=0 python run_cls_tnews.py   # 直接做了验证
