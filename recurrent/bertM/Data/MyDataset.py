import torch.utils.data as data
import pandas as pd
from tcn_test.data_tcn.parameters import Parameters

parameters = Parameters()


class MyDataset(data.Dataset):
    """
    必须继承data.Dataset类
    """

    def __init__(self, df: pd.DataFrame):
        """
        在这里进行初始化，一般是初始化文件路径或文件列表
        """
        self.df = df
        self.length = 0
        self.read_index = []
        self.window = parameters.Sliding_window_radius * 2 + 1
        self.init()

    def __getitem__(self, index):
        """
        1. 按照index，读取文件中对应的数据  （读取一个数据！！！！我们常读取的数据是图片，一般我们送入模型的数据成批的，但在这里只是读取一张图片，成批后面会说到）
        2. 对读取到的数据进行数据增强 (数据增强是深度学习中经常用到的，可以提高模型的泛化能力)
        3. 返回数据对 （一般我们要返回 图片，对应的标签） 在这里因为我没有写完整的代码，返回值用 0 代替

        :returns labels, sequence ：一个滑动窗口的标签和数据
        """
        # labels = []
        # sequence = []
        # for i in range(index, index + parameters.Sliding_window_radius * 2 + 1):
        #     labels.append(self.df.loc[i, 'Label'])
        #     sequence.append(self.df.loc[i, 'Action'])
        # return labels, sequence
        sequence = []
        for i in range(self.window):
            sequence.append(self.df.loc[self.read_index[index] + i, 'Action'])
        label = self.df['Label'][self.read_index[index] + self.window - 1]
        return label, sequence

    def __len__(self):
        """
        返回数据集的长度

        :return 滑动窗口的个数
        """
        # return self.df.__len__() - parameters.Sliding_window_radius * 2
        return self.length

    def init(self):
        grouped_data = self.df.groupby(['NewName', 'LevelNumber'])
        for name, group in grouped_data:
            for index in group.index:
                if group['DataCode'][index] == 5000:
                    df_cell = group.loc[(index - self.window + 1): index]
                    if len(df_cell) < self.window:
                        continue
                    self.length += 1
                    self.read_index.append(index - self.window + 1)
