import os
import random
import sys
import logging
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import torch
from torch import nn
from transformers import AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def init(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data(train_dir: str,
              test_dir: str,
              tokenizer,
              wrapped_tokenizer,
              WrapperClass,
              myTemplate):
    """
    数据为csv文件，包含两列Text和Label
    :param wrapped_tokenizer:
    :param train_dir:
    :param test_dir:
    :return:
    """
    # 准备数据集
    # dir_train = os.path.join(os.path.dirname(__file__), train_dir)
    # dir_test = os.path.join(os.path.dirname(__file__), test_dir)
    df_train = pd.read_csv(train_dir)
    df_test = pd.read_csv(test_dir)

    x_train, y_train = df_train['Text'].astype('str').to_numpy(), df_train['Label'].to_numpy()
    x_test, y_test = df_test['Text'].astype('str').to_numpy(), df_test['Label'].to_numpy()

    dataset = {'train': [], 'test': []}
    for index in range(len(x_train)):
        input_example = InputExample(text_a=x_train[index], label=int(y_train[index]), guid=index)
        dataset['train'].append(input_example)
    for index in range(len(x_test)):
        input_example = InputExample(text_a=x_test[index], label=int(y_test[index]), guid=index)
        dataset['test'].append(input_example)

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
    return train_dataloader, test_dataloader


def get_logger(name: str, level=logging.INFO, log_path='./log'):
    log = logging.getLogger(name)
    if log.handlers:
        log.setLevel(level)
        return log
    log.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if not os.path.exists(log_path):  # 如果路径不存在
        os.makedirs(log_path)
    file = logging.FileHandler(log_path + '/log.txt')
    file.setLevel(logging.INFO)
    file.setFormatter(formatter)
    log.addHandler(file)

    return log
