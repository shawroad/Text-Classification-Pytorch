# 中文文本分类模型集锦
## 数据说明
数据来源: https://github.com/CLUEbenchmark/CLUE

1. tnews_public数据(短文本分类)

该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。  
数据量：训练集(53,360)，验证集(10,000)，测试集(10,000)

2. iflytek_public数据(长文本分类) 

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。  
数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)

## 实验结果: 
| 模型 | iflytek(各类应用主题数据 长文本) Accuracy | tnews(今日头条新闻数据集 短文本) Accuracy | Avg |
| :-: | :-: | :-: | :-: | 
| FastText | 53.09 | 51.03 | 52.06 | 
| Roberta-base | 60.75 | 57.27 | 59.01 | 
| NEZHA-base | **?** | **0.582300 ** | **?** | 
| MengZi-base | 60.44 | 57.40  | 58.92 | 

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

