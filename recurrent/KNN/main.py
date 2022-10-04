import os

import pandas as pd
import numpy as np
import gensim
from m import BOW
import copy
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier

embedding_dim = 300
use_pretrained_embedding = True

dir_train = os.path.join(os.path.dirname(__file__), '../../data/single-sentence-prediction/train_data.csv')
dir_test = os.path.join(os.path.dirname(__file__), '../../data/single-sentence-prediction/test_data.csv')
train_data = pd.read_csv(dir_train)
test_data = pd.read_csv(dir_test)

x = train_data['Text'].append(test_data['Text'], ignore_index=True)
y = train_data['Label'].append(test_data['Label'], ignore_index=True)
bow = BOW(x.tolist(), min_count=1, maxlen=20)
vocab_size = len(bow.word2idx)
word2vec = gensim.models.KeyedVectors.load_word2vec_format('./wiki-news-300d-1M.vec', encoding='utf-8')
embedding_matrix = np.zeros((vocab_size + 1, 300))
for key, value in bow.word2idx.items():
    if key in word2vec.key_to_index:
        embedding_matrix[value] = word2vec.get_vector(key)
    else:
        embedding_matrix[value] = [0] * embedding_dim
X = copy.deepcopy(bow.doc2num)
X = torch.LongTensor(X)


class Embed_Layer(nn.Module):
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300):
        super(Embed_Layer, self).__init__()
        self.encoder = nn.Embedding(vocab_size + 1, embedding_dim)
        if use_pretrained_embedding:
            self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))

    def forward(self, x):
        """
        outputs:
        torch.Size([15806, 20])
        torch.Size([15806, 20, 300])
        torch.Size([15806, 6000])
        """
        x = self.encoder(x)
        x = x.view(len(x), 6000)
        return x


model = Embed_Layer(embedding_matrix=embedding_matrix, vocab_size=vocab_size)
X = model(X)
X = X.detach().numpy()
train_x = X[:len(train_data)]
train_y = train_data['Label']
test_x = X[len(train_data):]
test_y = test_data['Label']
clf = KNeighborsClassifier()
clf.fit(train_x, train_y)
pre = clf.predict(test_x)
print('test accuracy: ', accuracy_score(test_y, pre))
print('test kappa: ', cohen_kappa_score(test_y, pre))
print(classification_report(test_y, pre))
