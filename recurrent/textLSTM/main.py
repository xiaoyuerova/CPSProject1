from textLSTM import *
import pandas as pd
import numpy as np
import gensim
from m import BOW
import copy
import torch.utils.data as Data
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('../data/train_data.csv')
test_data = pd.read_csv('../data/test_data.csv')
x = train_data['Text'].append(test_data['Text'], ignore_index=True)
y = train_data['Label'].append(test_data['Label'], ignore_index=True)
bow = BOW(x.tolist(), min_count=1, maxlen=20)
vocab_size = len(bow.word2idx)
word2vec = gensim.models.KeyedVectors.load_word2vec_format('../wiki-news-300d-1M.vec', encoding='utf-8')
embedding_matrix = np.zeros((vocab_size + 1, 300))
for key, value in bow.word2idx.items():
    if key in word2vec.key_to_index:
        embedding_matrix[value] = word2vec.get_vector(key)
    else:
        embedding_matrix[value] = [0] * embedding_dim
X = copy.deepcopy(bow.doc2num)
BATCH_SIZE = 64
content_tensor = torch.from_numpy(np.array(X)).long()
label_tensor = torch.from_numpy(np.array(y)).float()
x_train = content_tensor[:12931]
y_train = label_tensor[:12931]
x_test = content_tensor[12931:]
y_test = label_tensor[12931:]
torch_dataset = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=BATCH_SIZE,  # mini batch size
    shuffle=False,  # random shuffle for training
    num_workers=0,  # subprocesses for loading data
)
# 网络结构、损失函数、优化器初始化
model = textLSTM(embedding_matrix, vocab_size)  # 加载预训练embedding matrix
# loss_func = nn.CrossEntropyLoss()
loss_func = nn.CrossEntropyLoss()
if USE_CUDA:
    model = model.cuda()  # 把搭建的网络载入GPU
    loss_func.cuda()  # 把损失函数载入GPU
optimizer = Adam(model.parameters(), lr=LR)  # 默认lr

# 开始跑模型
EPOCH = 10
for epoch in tqdm(range(EPOCH)):
    for batch_id, (data, target) in enumerate(train_loader):
        if USE_CUDA:
            data, target = data.cuda(), target.cuda()  # 数据载入GPU
        output = model(data)
        target = target.to(dtype=torch.int64)
        loss = loss_func(output, target)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
model.eval()
output = model(x_test.cuda())
label_test = y_test.to(dtype=torch.int64)
print('acc in test: ',
      accuracy_score(label_test.data.numpy(), np.argmax(output.cpu().data.numpy(), axis=1)))
