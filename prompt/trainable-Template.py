import os
import random
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
from transformers import AdamW, logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report


random.seed(0)
logging.set_verbosity_warning()
logging.set_verbosity_error()


# 准备数据集
dir_train = os.path.join(os.path.dirname(__file__), '../data/single-sentence-prediction/train_data.csv')
dir_test = os.path.join(os.path.dirname(__file__), '../data/single-sentence-prediction/test_data.csv')
df_train = pd.read_csv(dir_train)
df_test = pd.read_csv(dir_test)

x_train, y_train = df_train['Text'].astype('str').to_numpy(), df_train['Label'].to_numpy()
x_test, y_test = df_test['Text'].astype('str').to_numpy(), df_test['Label'].to_numpy()


dataset = {'train': [], 'test': []}
for index in range(len(x_train)):
    input_example = InputExample(text_a=x_train[index], label=int(y_train[index]), guid=index)
    dataset['train'].append(input_example)
for index in range(len(x_test)):
    input_example = InputExample(text_a=x_test[index], label=int(y_test[index]), guid=index)
    dataset['test'].append(input_example)


plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
special_tokens = {'additional_special_tokens': ['[R_zero]', '[R_one]', '[R_two]', '[R_three]',
                                                '[voltage]', '[current]', '[number]']}
tokenizer.add_special_tokens(special_tokens)        # 添加special tokens
plm.resize_token_embeddings(len(tokenizer))
myTemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"soft"} {"mask"}.')
wrapped_tokenizer = WrapperClass(max_seq_length=16, decoder_max_length=3, tokenizer=tokenizer,
                                 truncate_method="head")
model_inputs = {}
for split in ['train', 'test']:
    model_inputs[split] = []
    for sample in dataset[split]:
        tokenized_example = wrapped_tokenizer.tokenize_one_example(myTemplate.wrap_one_example(sample),
                                                                   teacher_forcing=False)
        model_inputs[split].append(tokenized_example)
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=myTemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=16, decoder_max_length=3,
                                    batch_size=64, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="head")
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=myTemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=16, decoder_max_length=3,
                                   batch_size=64, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")


myVerbalizer = ManualVerbalizer(tokenizer, num_classes=8,
                                label_words=[
                                    ["statement", "request", "understand", "need"],
                                    ["maintain", "communication"],
                                    ["negotiate"],
                                    ["share", "information"],
                                    ["formulation"],
                                    ["plan", "scheme", "strategy"],
                                    ["chats"],
                                    ["monitor"]
                                ])


use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=myTemplate, verbalizer=myVerbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()
loss_func = torch.nn.CrossEntropyLoss()
# it's always good practice to set no decay to biase and LayerNorm parameters
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)


# 训练
x, y , y1 = [], [], []
for epoch in range(6):
    tot_loss = 0
    allPred = []
    allLabel = []
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        allLabel.extend(labels.cpu().tolist())
        allPred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 99 or step == len(train_dataloader)-1:
            print("Epoch {}, step {}, average loss: {}".format(epoch + 1, step, tot_loss / (step + 1)), flush=True)
            accuracy = accuracy_score(allPred, allLabel)
            print('train accuracy: ', accuracy)
    y1.append(accuracy_score(allPred, allLabel))

    # 测试
    prompt_model.eval()
    allPred = []
    allLabel = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logit = prompt_model(inputs)
        labels = inputs['label']
        allLabel.extend(labels.cpu().tolist())
        allPred.extend(torch.argmax(logit, dim=-1).cpu().tolist())
    accuracy = accuracy_score(allPred, allLabel)
    print('Epoch {} test'.format(epoch + 1))
    print('test acc: ', accuracy)
    print('test kappa: ', cohen_kappa_score(allPred, allLabel))
    print(classification_report(allLabel, allPred))

    x.append(epoch + 1)
    y.append(accuracy)
df = pd.DataFrame({'epoch': x, 'accuracy': y, 'train_accuracy': y1})
print(df)
path = os.path.join(os.path.dirname(__file__), '../outputs/{}'.format(__file__[2:].replace('\\', '_')))
if not os.path.exists(path):
    os.makedirs(path)
df.to_csv(path + '\\' + os.path.basename(__file__)[:-3] + '-2', index=False)
