import torch
import torch.nn as nn

USE_CUDA = True
embedding_dim = 300
use_pretrained_embedding = True
LR = 0.001


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x, dropout_p=0.25):
        return nn.Dropout(p=dropout_p)(self.encoder(x))


class textGRU(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None):
        super(textGRU, self).__init__()
        self.embed_layer = Embed_Layer(embedding_matrix, vocab_size)
        self.encoder = nn.GRU(input_size=embedding_dim,
                              hidden_size=embedding_dim,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)
        self.decoder = nn.Linear(embedding_dim * 2, 11)
        self.FC = nn.Linear(embedding_dim, 11)

    def forward(self, x):
        x = self.embed_layer(x)
        x, _ = self.encoder(x)  # output, (h, c)
        x = self.decoder(x[:, -1, :])
        return x
