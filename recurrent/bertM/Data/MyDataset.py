import torch.utils.data as data
import pandas as pd


class MyDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame):
        df.reset_index(inplace=True)
        self.df = df

    def __getitem__(self, index):
        sequence = self.df['Text'][index]
        label = self.df['Label'][index]
        return label, sequence

    def __len__(self):
        return self.df.__len__()
