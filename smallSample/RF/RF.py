import json
import os
from Utils import init_outputs_dir, PROJECT_NAME, load_data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

# 输出
outputs = pd.DataFrame(columns=['Round', 'Epoch', 'Accuracy', 'Kappa'])
cr_results = []  # classification_report输出结果保存


def evaluate(evaluate_round: int, prop: str, dir_num=None):
    # 准备数据集
    train_data, test_data = load_data(prop, dir_num=dir_num)

    tfidf = TfidfVectorizer(norm='l2', max_features=10000, use_idf=True, smooth_idf=True, sublinear_tf=False,
                            ngram_range=(1, 3))
    tfidf.fit_transform(train_data['Text'])
    train_x = tfidf.transform(train_data['Text'])
    test_x = tfidf.transform(test_data['Text'])
    train_y = train_data['Label'].to_numpy()
    test_y = test_data['Label'].to_numpy()

    # model
    clf = RandomForestClassifier()
    # train
    clf.fit(train_x, train_y)
    # test
    pre = clf.predict(test_x)
    accuracy = accuracy_score(test_y, pre)
    kappa = cohen_kappa_score(test_y, pre)
    print('acc in test: ', accuracy)
    print('kappa in test: ', kappa)
    print(classification_report(test_y, pre))

    # 保存输出结果
    epoch = 1  # 记录epoch=1，方便统一统计结果
    outputs.loc[len(outputs)] = [evaluate_round + 1, epoch, accuracy, kappa]
    cr_results.append({'evaluate_round': evaluate_round,
                       'epoch': epoch,
                       'cr_result': classification_report(test_y, pre, output_dict=True)})


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
