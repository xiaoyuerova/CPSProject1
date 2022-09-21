import torch.nn as nn
from transformers import logging
from transformers import BertModel

logging.set_verbosity_warning()
logging.set_verbosity_error()


class CpsBertModel(nn.Module):
    def __init__(self):
        super(CpsBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(768, 8)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()

    def forward(self, _input):
        pooler_output = self.bert(**_input)[1]
        y = self.decoder(pooler_output)
        return y
