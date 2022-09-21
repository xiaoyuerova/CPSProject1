import time
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score, classification_report
import torch.nn as nn

from Data import *
from model import CpsBertModel
from parameters import parameters

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()

# 准备模型
model = CpsBertModel()

total = 0
total2 = 0
for param in model.parameters():
    total += param.nelement()
    if param.requires_grad:
        total2 += param.nelement()
print("Number of parameter: %.2fM" % (total / 1e6))
print("Number of training parameter: %.2fM" % (total2 / 1e6))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)


def evaluate(data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    y_pred, y_true = [], []
    start_time = time.time()
    model.eval()
    for idx, (label, text) in enumerate(data.dataloader):
        optimizer.zero_grad()
        output = model(text)

        total += label.size(0)

        loss = criterion(output, label)
        total_loss += loss.item()

        predict = output.argmax(1)
        for i in predict.eq(label):
            if i:
                correct += 1

#         y_pred.extend(predict.to('cpu'))
#         y_true.extend(label.to('cpu'))

    batches = data.dataloader.__len__()
    cur_loss = total_loss / batches
    elapsed = time.time() - start_time
    kappa = cohen_kappa_score(y_pred, y_true)
    print(
        '| epoch {:3d} | {:5d} batches | ms/batch {:5.5f} | loss {:5.2f} | '
        'accuracy {:8.2f}% | Kappa {:8.4f}'.format(
            epoch + 1, batches,
            elapsed * 1000 / batches, cur_loss,
            correct / total * 100,
            kappa))
    print(classification_report(y_true, y_pred, target_names=parameters.all_names))
    return correct / total


def train(train_data: Data, epoch: int):
    total_loss = 0
    correct = 0.0
    total = 0.0
    start_time = time.time()
    model.train()
    for idx, (label, text) in enumerate(train_data.dataloader):
        optimizer.zero_grad()
        output = model(text)

        total += label.size(0)
        for i in output.argmax(1).eq(label):
            if i:
                correct += 1

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        log_interval = parameters.log_interval
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | accuracy {:8.2f}%'.format(epoch + 1, idx, train_data.dataloader.__len__(),
                                                            optimizer.param_groups[0]['lr'],
                                                            elapsed * 1000 / log_interval, cur_loss,
                                                            correct / total * 100))
            total_loss = 0
            start_time = time.time()


def main():
    df_train = pd.read_csv('../data/single-sentence-prediction/train_data.csv')
    df_test = pd.read_csv('../data/single-sentence-prediction/test_data.csv')

    # 初始化数据
    dataset_train = MyDataset(df_train)
    data_train = Data(dataset_train)

    dataset_test = MyDataset(df_test)
    data_test = Data(dataset_test)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    for epoch in range(10):
        train(data_train, epoch)
        accu_val = evaluate(data_test, epoch)
        if total_accu is not None and total_accu > accu_val:
            print('scheduler runs')
            scheduler.step()
        else:
            total_accu = accu_val
    return model


if __name__ == '__main__':
    main()
