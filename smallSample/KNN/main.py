import json
import os

import pandas as pd
import numpy as np
import gensim
from smallSample.KNN.m import BOW
import copy
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from Utils import print_model, init_outputs_dir, load_data

embedding_dim = 300
use_pretrained_embedding = True
# 输出
outputs = pd.DataFrame(columns=['Round', 'Epoch', 'Accuracy', 'Kappa'])
cr_results = []  # classification_report输出结果保存


class EmbedLayer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(EmbedLayer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x):
        """
        outputs:
        torch.Size([15806, 20])
        torch.Size([15806, 20, 300])
        torch.Size([15806, 6000])
        """
        x = self.encoder(x)
        x = x.view(len(x), 6000)
        return x


def evaluate(evaluate_round: int, prop: str, dir_num=None):
    # 准备数据集
    train_data, test_data = load_data(prop, dir_num=dir_num)

    x = train_data['Text'].append(test_data['Text'], ignore_index=True)
    y = train_data['Label'].append(test_data['Label'], ignore_index=True)
    bow = BOW(x.tolist(), min_count=1, maxlen=20)
    vocab_size = len(bow.word2idx)
    src_dir = os.path.join(os.path.dirname(__file__), '../../data/src/wiki-news-300d-1M.vec')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(src_dir, encoding='utf-8')
    embedding_matrix = np.zeros((vocab_size + 1, 300))
    for key, value in bow.word2idx.items():
        if key in word2vec.key_to_index:
            embedding_matrix[value] = word2vec.get_vector(key)
        else:
            embedding_matrix[value] = [0] * embedding_dim
    X = copy.deepcopy(bow.doc2num)
    X = torch.LongTensor(X)

    # 准备模型
    model = EmbedLayer(embedding_matrix=embedding_matrix, vocab_size=vocab_size)
    print_model(model)

    # 开始训练
    epoch = 1  # KNN没必要训练多轮。记录epoch=1，方便统一统计结果
    X = model(X)
    X = X.detach().numpy()
    train_x = X[:len(train_data)]
    train_y = train_data['Label']
    test_x = X[len(train_data):]
    test_y = test_data['Label']
    clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    pre = clf.predict(test_x)
    accuracy = accuracy_score(test_y, pre)
    kappa = cohen_kappa_score(test_y, pre)
    print('test accuracy: ', accuracy)
    print('test kappa: ', kappa)
    print(classification_report(test_y, pre))

    # 保存输出结果
    outputs.loc[len(outputs)] = [evaluate_round + 1, epoch, accuracy, kappa]  # 不需要多轮训练，epoch值给个固定的1
    cr_results.append({'evaluate_round': evaluate_round,
                       'epoch': epoch,
                       'cr_result': classification_report(test_y, pre, output_dict=True)})


def main(numbers=None, props=None, dir_num=None):
    """
    运行前需要确定数据集。更改数据集后要注意修改以下几个参数：
    :param numbers: 输出保存文件的编号
        numbers = ['3-2']
    :param props: 数据在训练集中的比例
        如：props = ['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p']
    :param dir_num: 数据集所在位置small-sample-datasets（None对应的默认结果）等
        如：dir_num = 2，对应small-sample-datasets2
    :return:
    """
    if props is None:
        props = []
    if numbers is None:
        numbers = []

    for idx, number in enumerate(numbers):
        # 清空
        outputs.drop(outputs.index, inplace=True)
        cr_results = []

        # 按不同的数据跑模型
        for i in range(len(props)):
            print('**第' + str(i + 1) + '轮**')
            evaluate(i, props[i], dir_num=dir_num[idx])
        print(outputs)

        # 初始化输出文件的路径
        outputs_dir = init_outputs_dir(__file__)

        # 保存accuracy和kappa
        outputs.to_csv(outputs_dir + r'/' + os.path.basename(__file__)[:-3] + '{}.csv'.format(number), index=False)

        # 保存classification_report的输出结果
        f = open(outputs_dir + r'/' + os.path.basename(__file__)[:-3] + '{}.json'.format(number), 'w')
        f.write(json.dumps(cr_results))
        f.close()


if __name__ == '__main__':
    main(
        numbers=['3-t'],
        props=['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p'],
        dir_num=None
    )
