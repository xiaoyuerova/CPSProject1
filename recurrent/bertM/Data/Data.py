import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

special_tokens = {'additional_special_tokens': ['[R_zero]', '[R_one]', '[R_two]', '[R_three]',
                                                '[voltage]', '[current]', '[number]']}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens(special_tokens)


class Data:
    def __init__(self, dataset):
        self.label_pipeline = lambda x: [int(item) for item in x]
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        label_list, text_list = [], []
        for (labels, texts) in batch:
            label_list.append(self.label_pipeline(labels))
            text_list.append(self.tokenizer(texts, return_tensors='pt'))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        return label_list, text_list
