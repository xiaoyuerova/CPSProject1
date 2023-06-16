from torch.utils.data import Dataset
import pandas as pd
from openprompt.data_utils import InputExample
from .Config import Config


class MyDataset(Dataset):
    def __init__(self, data_dir: str, args: Config):
        self.data_dir = data_dir
        self.class_to_index = {key: i for i, key in enumerate(args.classes)}
        self.src = self.get_src()

    def __getitem__(self, index):
        return self.src[index]

    def __len__(self):
        return len(self.src)

    def get_src(self):
        src = pd.read_csv(self.data_dir)
        src.reset_index(inplace=True)
        return [
            InputExample(text_a=src['Text'][index],
                         label=self.class_to_index[src['Label'][index]],
                         guid=index)
            for index in src.index
        ]


