import json
import os

from textLSTM import *
import pandas as pd
import numpy as np
import gensim
from m import BOW
import copy
import torch.utils.data as Data
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from Utils import init_outputs_dir, PROJECT_NAME, load_data

BATCH_SIZE = 256
EPOCH = 10

# 输出
outputs = pd.DataFrame(columns=['Round', 'Epoch', 'Accuracy', 'Kappa'])
cr_results = []  # classification_report输出结果保存


def evaluate(evaluate_round: int, prop: str, dir_num=None):
    # 准备数据集
    train_data, test_data = load_data(prop, dir_num=dir_num)

    x = train_data['Text'].append(test_data['Text'], ignore_index=True)
    y = train_data['Label'].append(test_data['Label'], ignore_index=True)
    bow = BOW(x.tolist(), min_count=1, maxlen=16)
    vocab_size = len(bow.word2idx)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format('../wiki-news-300d-1M.vec', encoding='utf-8')
    embedding_matrix = np.zeros((vocab_size + 1, 300))
    for key, value in bow.word2idx.items():
        if key in word2vec.key_to_index:
            embedding_matrix[value] = word2vec.get_vector(key)
        else:
            embedding_matrix[value] = [0] * embedding_dim
    X = copy.deepcopy(bow.doc2num)

    content_tensor = torch.from_numpy(np.array(X)).long()
    label_tensor = torch.from_numpy(np.array(y)).float()
    x_train = content_tensor[:len(train_data)]
    y_train = label_tensor[:len(train_data)]
    x_test = content_tensor[len(train_data):]
    y_test = label_tensor[len(train_data):]
    torch_dataset = Data.TensorDataset(x_train, y_train)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=0,  # subprocesses for loading data
    )
    # 网络结构、损失函数、优化器初始化
    model = textLSTM(embedding_matrix, vocab_size)  # 加载预训练embedding matrix
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.CrossEntropyLoss()
    if USE_CUDA:
        model = model.cuda()  # 把搭建的网络载入GPU
        loss_func.cuda()  # 把损失函数载入GPU
    optimizer = Adam(model.parameters(), lr=LR)  # 默认lr

    # 开始跑模型
    for epoch in tqdm(range(EPOCH)):
        model.train()
        for batch_id, (data, target) in enumerate(train_loader):
            if USE_CUDA:
                data, target = data.cuda(), target.cuda()  # 数据载入GPU
            output = model(data)
            target = target.to(dtype=torch.int64)
            loss = loss_func(output, target)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        model.eval()
        output = model(x_test.cuda())
        label_test = y_test.to(dtype=torch.int64)

        all_label = label_test.data.numpy()
        all_pre = np.argmax(output.cpu().data.numpy(), axis=1)
        accuracy = accuracy_score(all_label, all_pre)
        kappa = cohen_kappa_score(all_label, all_pre)
        print('acc in test: ', accuracy)
        print('kappa in test: ', kappa)
        print(classification_report(all_label, all_pre))

        # 保存输出结果
        outputs.loc[len(outputs)] = [evaluate_round + 1, epoch, accuracy, kappa]
        cr_results.append({'evaluate_round': evaluate_round,
                           'epoch': epoch,
                           'cr_result': classification_report(all_label, all_pre, output_dict=True)})


def main(numbers=None, props=None, dir_num=None):
    """
    运行前需要确定数据集。更改数据集后要注意修改以下几个参数：
    :param numbers: 输出保存文件的编号
        numbers = ['3-2']
    :param props: 数据集所在位置small-sample-datasets（None对应的默认结果）等
        如：dir_num = 2，对应small-sample-datasets2
    :param dir_num: 数据在训练集中的比例
        如：props = ['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p']
    :return:
    """
    for number in numbers:
        if props is None:
            props = []
        # 按不同的数据跑模型
        for i in range(len(props)):
            print('**第' + str(i + 1) + '轮**')
            evaluate(i, props[i], dir_num=dir_num)
        print(outputs)

        # 初始化输出文件的路径
        outputs_dir = init_outputs_dir(
            __file__[__file__.find(PROJECT_NAME) + len(PROJECT_NAME) + 1:-3].replace('/', '-'))

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
