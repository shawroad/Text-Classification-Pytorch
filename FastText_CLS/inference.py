import fasttext.FastText as fasttext
from tqdm import tqdm


if __name__ == '__main__':
    model_save_path = './output/fasttext_cls.model'
    model = fasttext.load_model(model_save_path)
    
    valid_data_path = './data/valid.txt' 
    
    fenmu, fenzi = 0, 0
    with open(valid_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            fenmu += 1
            label, sentence = line.strip().split(',', 1)   # 在第一个逗号处一分为二
            label, sentence = label.strip(), sentence.strip()
            result = model.predict(sentence)   # k=3 可以返回top3的结果
            predict_label = result[0][0].strip()
            
            if label == predict_label:
                fenzi += 1
    acc = fenzi / fenmu 
    score = model.test(valid_data_path)
    print('valid_size:{}, Precision:{}, Recall:{}, Accuracy:{}'.format(score[0], score[1], score[2], acc))
    
