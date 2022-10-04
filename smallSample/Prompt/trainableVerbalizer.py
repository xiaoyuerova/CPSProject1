import json
import math
import os
import random
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import SoftVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
from transformers import AdamW, logging
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

from Utils import PROJECT_NAME, init_outputs_dir, load_data

random.seed(0)
logging.set_verbosity_warning()
logging.set_verbosity_error()

BACH_SIZE = 128

# 输出
outputs = pd.DataFrame(columns=['Round', 'Epoch', 'Accuracy', 'Kappa'])
cr_results = []  # classification_report输出结果保存


def print_model(model: torch.nn.Module):
    total = 0
    total2 = 0
    for param in model.parameters():
        total += param.nelement()
        if param.requires_grad:
            total2 += param.nelement()
    print("Number of parameter: %.2fM" % (total / 1e6))
    print("Number of training parameter: %.2fM" % (total2 / 1e6))


def evaluate(evaluate_round: int, prop: str, dir_num=None):
    # 准备数据集
    df_train, df_test = load_data(prop, dir_num=dir_num)

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
    tokenizer.add_special_tokens(special_tokens)  # 添加special tokens
    plm.resize_token_embeddings(len(tokenizer))
    template_text = '{"placeholder":"text_a"} It was {"mask"}.'
    myTemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
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
                                        batch_size=BACH_SIZE, shuffle=True, teacher_forcing=False,
                                        predict_eos_token=False,
                                        truncate_method="head")
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=myTemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=16, decoder_max_length=3,
                                       batch_size=BACH_SIZE, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False,
                                       truncate_method="head")

    myVerbalizer = SoftVerbalizer(tokenizer, plm, num_classes=8)
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
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    print_model(prompt_model)
    # 训练
    x, y, y1 = [], [], []
    for epoch in range(5):
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
            if step % 100 == 99 or step == len(train_dataloader) - 1:
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
        kappa = cohen_kappa_score(allPred, allLabel)
        print('Epoch {} test'.format(epoch + 1))
        print('test acc: ', accuracy)
        print('test kappa: ', kappa)
        print(classification_report(allLabel, allPred))

        # 保存输出结果
        outputs.loc[len(outputs)] = [evaluate_round + 1, epoch, accuracy, kappa]
        cr_results.append({'evaluate_round': evaluate_round,
                           'epoch': epoch,
                           'cr_result': classification_report(allLabel, allPred, output_dict=True)})

        x.append(epoch + 1)
        y.append(accuracy)
    df = pd.DataFrame({'epoch': x, 'accuracy': y, 'train_accuracy': y1})
    print(df)


def main(numbers=None, props=None, dir_num=None):
    """
    运行前需要确定数据集。更改数据集后要注意修改以下几个参数：
    :param numbers: 输出保存文件的编号
        numbers = ['3-2']
    :param props: 数据集所在位置small-sample-datasets（None对应的默认结果）等
        如：dir_num = 2，对应small-sample-datasets2
    :param dir_num: 数据在训练集中的比例
        如：props = ['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p']
    :return:
    """
    for number in numbers:
        if props is None:
            props = []
        for i in range(len(props)):
            print('**第' + str(i + 1) + '轮**')
            evaluate(i, props[i], dir_num=dir_num)
        print(outputs)

        # 初始化输出文件的路径
        outputs_dir = init_outputs_dir(
            __file__[__file__.find(PROJECT_NAME) + len(PROJECT_NAME) + 1:-3].replace('/', '-'))

        # 保存accuracy和kappa
        outputs.to_csv(outputs_dir + r'/' + os.path.basename(__file__)[:-3] + '{}.csv'.format(number), index=False)

        # 保存classification_report的输出结果
        f = open(outputs_dir + r'/' + os.path.basename(__file__)[:-3] + '{}.json'.format(number), 'w')
        f.write(json.dumps(cr_results))
        f.close()


if __name__ == '__main__':
    main(
        numbers=['3-t'],
        props=['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p'],
        dir_num=None
    )
