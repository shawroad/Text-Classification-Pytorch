import json
import jieba
from tqdm import tqdm


def load_stopwords():
    path = '../data/stopwords.txt'
    stopwords = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            stopwords.append(line.strip())
    return stopwords


def load_data_and_clean(source_path, target_path):
    sentences = []
    with open(source_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = json.loads(line.strip())
            # 'label_des': '薅羊毛', 'sentence'
            label = line['label_des']
            sentence = line['sentence']
            
            # 分词 去停用词以及出现一次的词
            sentence_seg = jieba.lcut(sentence)

            # sentence_seg = filter(lambda x: len(x) > 1, sentence_seg)   # 去除低频词

            sentence_seg = filter(lambda x: x not in stopwords, sentence_seg)   # 去掉停用词
            
            sentences.append('__label__' + str(label) + ' , ' + ' '.join(sentence_seg))

    # 将数据写入
    with open(target_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(sentences))
    return sentences


if __name__ == '__main__':
    train_path = '../data/iflytek_public/train.json'
    save_processed_train_path = './data/train.txt'
    valid_path = '../data/iflytek_public/dev.json'
    save_processed_valid_path = './data/valid.txt'
    
    
    # 1. 加载停用词
    stopwords = load_stopwords()
    
    # 2. 将数据处理成fasttext能处理的格式 
    print('开始处理训练集...')
    load_data_and_clean(train_path, save_processed_train_path)
    print('训练集处理完毕...')

    print('开始处理验证集...')
    load_data_and_clean(valid_path, save_processed_valid_path)
    print('验证集处理完毕...')

    
    
    
