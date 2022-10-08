import os
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_NAME = os.path.basename(os.path.dirname(os.path.dirname(__file__)))


def load_data(prop: str, dir_num=None):
    if dir_num is None:
        base_path = os.path.join(os.path.dirname(__file__), '../data/small-sample-datasets/')
    else:
        base_path = os.path.join(os.path.dirname(__file__), '../data/small-sample-datasets{}/'.format(dir_num))
    dir_train = base_path + 'small-sample-trainsets{}.csv'.format(prop)
    dir_test = base_path + 'small-sample-testsets.csv'
    train_data = pd.read_csv(dir_train)
    test_data = pd.read_csv(dir_test)
    return train_data, test_data


def print_model(model):
    total = 0
    total2 = 0
    for param in model.parameters():
        total += param.nelement()
        if param.requires_grad:
            total2 += param.nelement()
    print("Number of parameter: %.2fM" % (total / 1e6))
    print("Number of training parameter: %.2fM" % (total2 / 1e6))


def draw(x, y, file, x2=None, y2=None):
    plt.title(os.path.basename(file))
    # if x2 is None:
    plt.plot(x, y)
    # else:
    #     plt.plot(x, y, x2, y2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    path = os.path.join(os.path.dirname(__file__), '../outputs/{}'.format(file))
    plt.savefig(path)


def init_outputs_dir(file: str):
    out_name = file[file.find(PROJECT_NAME) + len(PROJECT_NAME) + 1:-3].replace('/', '-')
    outputs_dir = os.path.join(os.path.dirname(__file__), '../outputs/{}'.format(out_name))
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    outputs_dir = os.path.abspath(outputs_dir)
    print('outputs_dir: ', outputs_dir)
    return outputs_dir


if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(__file__), '../data/small-sample-datasets/')

