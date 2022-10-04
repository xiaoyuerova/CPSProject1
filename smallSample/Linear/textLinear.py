import torch
import torch.nn as nn

USE_CUDA = True
embedding_dim = 300
use_pretrained_embedding = True
BATCH_SIZE = 128
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28
LR = 0.001
T_epsilon = 1e-7
num_classes = 11
Ks = [2, 3, 4]


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)(self.encoder(x))


class FullNet(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(FullNet, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.fc = nn.Linear(300 * 20, 300)
        self.fc2 = nn.Linear(300, 11)

    def forward(self, x):
        x = self.embed_layer(x)
        x = x.view(-1, 300 * 20)
        x = self.fc(x)
        x = self.fc2(x)
        return x
