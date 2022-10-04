import math
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import random
import pandas as pd
import torch
import torch.nn as nn
from transformers import logging
from recurrent.bertM.Data import *
from recurrent.bertM.Model import *
from Utils import PROJECT_NAME, init_outputs_dir, load_data, print_model

random.seed(0)
logging.set_verbosity_warning()
logging.set_verbosity_error()

# 输出
outputs = pd.DataFrame(columns=['Round', 'Epoch', 'Accuracy', 'Kappa'])
cr_results = []  # classification_report输出结果保存


def evaluate(evaluate_round: int, prop: str, dir_num: str):
    # 准备模型
    model = CpsBertModel()
    model.to(parameters.device)
    print_model(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=parameters.lr)

    # 初始化数据
    data_train, data_test = load_data(prop, dir_num=dir_num)
    data_train = Data(MyDataset(data_train))
    data_test = Data(MyDataset(data_test))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    for epoch in range(parameters.epochs):
        train(epoch, data_train, model, criterion, optimizer)
        accu_val, kappa, cr_result = test(epoch, data_test, model, criterion, optimizer)

        # 保存输出结果
        outputs.loc[len(outputs)] = [evaluate_round + 1, epoch, accu_val, kappa]
        cr_results.append({'evaluate_round': evaluate_round,
                           'epoch': epoch,
                           'cr_result': cr_result})
        if total_accu is not None and total_accu > accu_val:
            print('scheduler runs')
            scheduler.step()
        else:
            total_accu = accu_val
    return model


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
    for number in numbers:
        for i in range(len(props)):
            print('**第' + str(i + 1) + '轮**')
            evaluate(i, props[i], dir_num)
        print(outputs)

        # 初始化输出文件的路径
        outputs_dir = init_outputs_dir(
            __file__[__file__.find(PROJECT_NAME) + len(PROJECT_NAME) + 1:-3].replace(r'/', '-'))

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
