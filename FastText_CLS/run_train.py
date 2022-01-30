import os
import argparse
import fasttext.FastText as fasttext 


def train_model():
    model = fasttext.train_supervised(args.train_data_path, lr=args.lr, dim=args.embedding_dim, minCount=1,
                                      loss=args.loss_func, label='__label__', wordNgrams=3, epoch=args.epoch)
    """
      训练一个监督模型, 返回一个模型对象
      @param input:           训练数据文件路径
      @param lr:              学习率
      @param dim:             向量维度
      @param ws:              cbow模型时使用
      @param epoch:           次数
      @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
      @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
      @param minn:            构造subword时最小char个数
      @param maxn:            构造subword时最大char个数
      @param neg:             负采样
      @param wordNgrams:      n-gram个数
      @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
      @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
      @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
      @param lrUpdateRate:    学习率更新
      @param t:               负采样阈值
      @param label:           类别前缀
      @param verbose:         ??
      @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
      @return model object
    """
    # 保存模型
    output_model_file = os.path.join(args.output, "fasttext_cls.model")
    model.save_model(output_model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('--fasttext进行文本分类')
    parser.add_argument('--train_data_path', default='./data/train.txt', type=str, help='训练数据集路径')
    parser.add_argument('--valid_data_path', default='./data/valid.txt', type=str, help='验证集路径')
    parser.add_argument('--embedding_dim', default=300, type=int, help='词嵌入维度')
    parser.add_argument('--loss_func', default='softmax', type=str, help='使用的loss函数')
    parser.add_argument('--epoch', default=100, type=int, help='训练几轮')
    parser.add_argument('--lr', default=0.51, type=float, help='学习率大小')
    parser.add_argument('--output', default='output', type=str, help='输出路径')
    args = parser.parse_args() 

    os.makedirs(args.output, exist_ok=True)
    train_model()
    

