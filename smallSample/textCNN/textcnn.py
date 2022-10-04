import torch
import torch.nn as nn
import torch.nn.functional as F

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
Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)(self.encoder(x))


class textCNN(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(textCNN, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, 2, (K, 300)) for K in Ks])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(len(Ks) * 2, 11)

    def forward(self, x):
        x = self.embed_layer(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        x = self.dropout(x)
        x = self.fc(x)
        return x
