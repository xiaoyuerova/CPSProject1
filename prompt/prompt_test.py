import logging
import os
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, MixedTemplate, SoftTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
import torch
import torch.utils.data as data
from torch.optim import AdamW
import pandas as pd

from Prompt import *
from Coach import Coach


def main(args: Config):
    base_log = get_logger(__name__)
    base_log.debug(args)
    init(args.seed)

    # 第一步：定义任务
    train_dataset = Dataset(data_dir=args.train_dir, args=args)
    total = train_dataset.__len__()
    s = int(70 / 85 * total)
    train_dataset, dev_dataset = data.random_split(
        dataset=train_dataset,
        lengths=[s, total - s],
        generator=torch.Generator().manual_seed(args.seed)
    )
    test_dataset = Dataset(data_dir=args.test_dir, args=args)

    # 第二步：选择预训练模型
    plm, tokenizer, model_config, WrapperClass = load_plm(args.premodel_name, args.premodel_path)
    if args.add_special_tokens:
        tokenizer.add_special_tokens(args.special_tokens)  # 添加special tokens
        plm.resize_token_embeddings(len(tokenizer))

    # 第三步：定义模板（Template）
    if args.manual_t:
        my_template = ManualTemplate(tokenizer=tokenizer, text=args.template_text_manual)
    else:
        my_template = MixedTemplate(model=plm, tokenizer=tokenizer, text=args.template_text_trained)

    # 第四步：定义映射（Verbalizer）
    if args.manual_v:
        my_verbalizer = ManualVerbalizer(
            tokenizer=tokenizer,
            classes=args.classes,
            label_words=args.label_words
        )
    else:
        my_verbalizer = SoftVerbalizer(tokenizer, plm, classes=args.classes)

    train_dataloader = PromptDataLoader(dataset=list(train_dataset), template=my_template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_sentence_length,
                                        decoder_max_length=3,
                                        batch_size=args.batch_size, shuffle=args.shuffle, teacher_forcing=False,
                                        predict_eos_token=False,
                                        truncate_method="head")
    dev_dataloader = PromptDataLoader(dataset=list(dev_dataset), template=my_template, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_sentence_length,
                                      decoder_max_length=3,
                                      batch_size=args.batch_size, shuffle=args.shuffle, teacher_forcing=False,
                                      predict_eos_token=False,
                                      truncate_method="head")
    test_dataloader = PromptDataLoader(dataset=list(test_dataset), template=my_template, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_sentence_length,
                                       decoder_max_length=3,
                                       batch_size=args.batch_size, shuffle=args.shuffle, teacher_forcing=False,
                                       predict_eos_token=False,
                                       truncate_method="head")

    # 实例化模型，设置优化器和损失函数
    prompt_model = PromptForClassification(plm=plm, template=my_template, verbalizer=my_verbalizer, freeze_plm=False)
    if args.use_cuda:
        prompt_model = prompt_model.to(args.device)

    # it's always good practice to set no decay to bias and LayerNorm parameters
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    coach = Coach(train_dataloader,
                  dev_dataloader,
                  test_dataloader,
                  model=prompt_model,
                  opt=optimizer,
                  args=args)
    if not args.from_begin:
        ckpt = torch.load(args.model_file)
        coach.load_ckpt(ckpt)

    ret = coach.train()
    if args.premodel_name in ['t5', 'gpt2']:
        checkpoint = {
            "best_dev_f1": ret[0],
            "best_epoch": ret[1],
            # "best_state": ret[2],
        }
    else:
        checkpoint = {
            "best_dev_f1": ret[0],
            "best_epoch": ret[1],
            "best_state": ret[2],
        }
    print('ret[0]', ret[0])
    try:
        torch.save(checkpoint, args.model_file)
    except Exception as e:
        print('{} save error!/n'.format(args_.prompt_bias + args_.premodel_name), e)
    return [ret[1]] + ret[3]


if __name__ == '__main__':
    prompt_bias = ['base', 'trainableTemplate', 'trainableTemplateVerbalizer', 'trainableVerbalizer']
    premodel_name = ['bert', 'roberta', 'gpt2', 't5']
    premodel_path = ['bert-base-cased', 'roberta-base', 'gpt2', 't5-base']

    # label_words 自动选择的时候不需要参数

    out = pd.DataFrame(columns=['prompt_bias', 'premodel', 'best_epoch', 'f1', 'acc', 'kappa', 'CRF_acc'])
    error_list = []
    for i in range(len(prompt_bias)):
        for j in range(len(premodel_name)):
            args_ = Config(
                from_begin=True,
                train_dir='../data/single-sentence-prediction/train_data_v2.csv',
                test_dir='../data/single-sentence-prediction/test_data_v2.csv',
                epochs=20,
                batch_size=16,
                lr=1e-5,
                seed=1,
                class_focus='CRF',
                add_special_tokens=True,
                prompt_bias=prompt_bias[i],
                premodel_name=premodel_name[j],
                premodel_path=premodel_path[j],
                device='cuda:1'
            )
            args_.model_file = './save/{}/{}.pt'.format(args_.seed, args_.prompt_bias + args_.premodel_name)
            try:
                out_item = main(args_)
                out.loc[len(out)] = [args_.prompt_bias, args_.premodel_name] + out_item
            except Exception as e:
                print(e)
                error_list.append(args_.model_file)

    for i in [1, 2]:
        j = 2
        args_ = Config(
            from_begin=True,
            train_dir='../data/single-sentence-prediction/train_data_v2.csv',
            test_dir='../data/single-sentence-prediction/test_data_v2.csv',
            epochs=20,
            batch_size=24,
            lr=1e-5,
            seed=1,
            class_focus='CRF',
            add_special_tokens=True,
            prompt_bias=prompt_bias[i],
            premodel_name=premodel_name[j],
            premodel_path=premodel_path[j],
            device='cuda:1',
            special_tokens={'additional_special_tokens': ['[R_zero]', '[R_one]', '[R_two]', '[R_three]',
                                                    '[voltage]', '[current]', '[number]',
                                                    '[soft1]', '[soft1]', '[soft3]', '[soft4]', '[soft5]']},
            template_text_trained='{"placeholder":"text_a"} [soft1] [soft2] [soft3] [soft4] {"mask"} [soft5]'
        )
        args_.model_file = './save/{}/{}.pt'.format(args_.seed, args_.prompt_bias + args_.premodel_name)
        try:
            out_item = main(args_)
            out.loc[len(out)] = [args_.prompt_bias, args_.premodel_name] + out_item
        except Exception as e:
            print(e)
            error_list.append(args_.model_file)
    print('error_list: ', error_list)
    out.to_csv('./save/{}/out.csv'.format(args_.seed), index=False)

