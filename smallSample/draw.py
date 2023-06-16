import json
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
from Utils import init_outputs_dir

# 初始化输出文件的路径
outputs_dir = init_outputs_dir(__file__)


def init_dir(dirs: dict):
    base_dir = os.path.join(os.path.dirname(__file__), '../outputs/{}/{}.csv')
    ret = {}
    for k, v in dirs.items():
        ret[k] = base_dir.format(v[0], v[1] + '{}')  # 路径类似这样：'../outputs/recurrent-bertM-test/test3-1.csv'
    return ret


def calculate(base_dir: str, column: str, numbers: list):
    df = pd.DataFrame()
    for idx, num in enumerate(numbers):
        df_t = pd.read_csv(base_dir.format(num))
        df.insert(idx, str(idx), df_t.groupby('Round')[column].max())
    return df.sum(axis=1) / len(df.columns)


# def draw(model_output_dir: dict, numbers: list, props: list, data_dir_num=None):
#     dirs = init_dir(model_output_dir)
#     print(dirs)
#     df = pd.DataFrame(
#         {k + '_Accuracy': calculate(v, 'Accuracy', numbers) for k, v in dirs.items()})
#     df['X'] = [item.replace('p', '%') for item in props]
#     plt.close('all')
#     plt.figure()
#     acc_columns = df.columns.tolist()
#     acc_columns.remove('X')
#     df.plot('X', acc_columns, xlabel='Proportion', title='Variation of Accuracy value')
#     file_name = 'accuracy值变化{}.png'.format(data_dir_num) if data_dir_num else 'accuracy值变化.png'
#     plt.savefig(outputs_dir + '/{}'.format(file_name), dpi=300)
#
#     df = pd.DataFrame(
#         {k + '_Kappa': calculate(v, 'Kappa', numbers) for k, v in dirs.items()})
#     df['X'] = [item.replace('p', '%') for item in props]
#     plt.close('all')
#     plt.figure()
#     acc_columns = df.columns.tolist()
#     acc_columns.remove('X')
#     df.plot('X', acc_columns, xlabel='Proportion', title='Variation of Kappa value')
#     file_name = 'kappa值变化{}.png'.format(data_dir_num) if data_dir_num else 'kappa值变化.png'
#     plt.savefig(outputs_dir + '/{}'.format(file_name), dpi=300)


def draw(model_output_dir: dict, numbers: list, props: list, data_dir_num=None):
    dirs = init_dir(model_output_dir)
    print(dirs)
    df = pd.DataFrame(
        {k + '_Accuracy': calculate(v, 'Accuracy', numbers) for k, v in dirs.items()})
    df['X'] = [item.replace('p', '%') for item in props]
    plt.close('all')
    plt.figure()
    acc_columns = df.columns.tolist()
    acc_columns.remove('X')
    plt.axhline(y=0.6, ls='--', c='red')
    for col in acc_columns:
        plt.plot(df['X'], df[col], 'o-', label=col)
    plt.xlabel('Proportion')
    plt.ylabel('Accuracy')
    plt.title('Variation of Accuracy value')
    plt.legend()
    file_name = 'accuracy值变化{}.png'.format(data_dir_num) if data_dir_num else 'accuracy值变化.png'
    plt.savefig(outputs_dir + '/{}'.format(file_name), dpi=300)

    df = pd.DataFrame(
        {k + '_Kappa': calculate(v, 'Kappa', numbers) for k, v in dirs.items()})
    df['X'] = [item.replace('p', '%') for item in props]
    plt.close('all')
    plt.figure()
    acc_columns = df.columns.tolist()
    acc_columns.remove('X')
    plt.axhline(y=0.6, ls='--', c='red')
    for col in acc_columns:
        plt.plot(df['X'], df[col], 'o-', label=col)
    plt.xlabel('Proportion')
    plt.ylabel('Kappa')
    plt.title('Variation of Kappa value')
    plt.legend()
    file_name = 'kappa值变化{}.png'.format(data_dir_num) if data_dir_num else 'kappa值变化.png'
    plt.savefig(outputs_dir + '/{}'.format(file_name), dpi=300)


def main(model_output_dir=None, numbers=None, props=None, data_dir_num=None):
    """
    使用[model_output_dir]中的模型，在按比例[props]划分的数据集[data_dir_num]上跑[numbers]轮次的结果，画图.

    :param model_output_dir: dict 模型的输出路径和文件名。
    :param numbers: list[str] 每个模型跑的轮次对应的编号
    :param props: list[str] 数据在训练集中的比例.
        如：props = ['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p']
    :param data_dir_num: str 会决定输出的文件名
        在使用了不同的数据划分方式时使用。数据集所在位置small-sample-datasets（None对应的默认结果），
        如：data_dir_num = 2，对应small-sample-datasets2，则输出的文件名为"accuracy值变化2.png"
    :return:
    """
    if model_output_dir is None:
        model_output_dir = []
        print('parameter model_output_dir is None!')
    if props is None:
        props = []
        print('parameter props is None!')
    if numbers is None:
        numbers = []
        print('parameter numbers is None!')
    draw(
        model_output_dir=model_output_dir,
        numbers=numbers,
        props=props,
        data_dir_num=data_dir_num
    )


if __name__ == '__main__':
    # for item in ['1', '2', '3', '4', '5']:
    #     main(
    #         model_output_dir={
    #             'Prompt': ['smallSample-Prompt-trainableVerbalizer', 'trainableVerbalizer'],
    #             'Bert': ['recurrent-bertM-test', 'test'],
    #             'KNN': ['smallSample-KNN-main', 'main'],
    #             'Linear': ['smallSample-Linear-main', 'main'],
    #             'RF': ['smallSample-RF-RF', 'RF'],
    #             'textCNN': ['smallSample-textCNN-main', 'main'],
    #             'textGRU': ['smallSample-textGRU-main', 'main'],
    #             'textLSTM': ['smallSample-textLSTM-main', 'main'],
    #         },
    #         numbers=[item],
    #         props=['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p'],
    #         data_dir_num=item
    #     )

    main(
        model_output_dir={
            'Prompt': ['smallSample-Prompt-trainableVerbalizer', 'trainableVerbalizer'],
            'Bert': ['recurrent-bertM-test', 'test'],
            'KNN': ['smallSample-KNN-main', 'main'],
            'Linear': ['smallSample-Linear-main', 'main'],
            'RF': ['smallSample-RF-RF', 'RF'],
            'textCNN': ['smallSample-textCNN-main', 'main'],
            'textGRU': ['smallSample-textGRU-main', 'main'],
            'textLSTM': ['smallSample-textLSTM-main', 'main'],
        },
        numbers=['kf1', 'kf2', 'kf3', 'kf4', 'kf5'],
        props=['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p'],
        data_dir_num=None,
    )
