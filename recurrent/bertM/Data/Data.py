import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from recurrent.bertM.Data.parameters import parameters

special_tokens = {'additional_special_tokens': ['[R_zero]', '[R_one]', '[R_two]', '[R_three]',
                                                '[voltage]', '[current]', '[number]']}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_special_tokens(special_tokens)


class Data:
    def __init__(self, dataset):
        self.label_pipeline = lambda x: [int(item) for item in x]
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(dataset, batch_size=parameters.Batch_size, shuffle=True, collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        labels, texts = [], []
        try:
            for (label, text) in batch:
                labels.append(label)
                texts.append(text)
            labels = self.label_pipeline(labels)
            inputs = self.tokenizer(texts,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=parameters.max_length,
                                    )
            return torch.tensor(labels, dtype=torch.int64).to(parameters.device), inputs.to(parameters.device)

        except:
            print('texts:', texts)
            for text in texts:
                print(type(text), text)
