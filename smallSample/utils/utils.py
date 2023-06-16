import os
import pandas as pd


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