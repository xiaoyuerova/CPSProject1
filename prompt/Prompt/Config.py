from typing import Optional

import torch

CLASS = ['SMC', 'SSI', 'SESU', 'SN', 'CRF', 'CP', 'CEC', 'CMC']
SPECIAL_TOKENS = {'additional_special_tokens': ['[R_zero]', '[R_one]', '[R_two]', '[R_three]',
                                                '[voltage]', '[current]', '[number]']}
PROMPT_BIAS = ['base', 'trainableTemplate', 'trainableTemplateVerbalizer', 'trainableVerbalizer']
LABEL_WORDS = {
    'SMC': ["statement", "request", "understand", "need"],
    'SSI': ["maintain", "communication"],
    'SESU': ["negotiate"],
    'SN': ["share", "information"],
    'CRF': ["formulation"],
    'CP': ["plan", "scheme", "strategy"],
    'CEC': ["chats"],
    'CMC': ["monitor"]
}
LOSS_WEIGHT = [4.983844911147012,
               1.0,
               1.8765206812652069,
               5.3698868581375105,
               17.380281690140844,
               5.7879924953095685,
               4.577151335311573,
               5.176174496644295]


class Config(object):
    def __init__(self,
                 # 模型参数
                 train_dir: str,
                 test_dir: str,
                 model_file: str = './save/model.pt',
                 prompt_bias: str = 'base',  # choose in PROMPT_BIAS
                 max_sentence_length: Optional[str] = 16,  # openprompt包里面写的str，感觉可能是写错了

                 # 训练参数
                 from_begin=True,
                 seed=0,
                 epochs=6,
                 batch_size=4,
                 lr=1e-4,
                 shuffle=True,
                 use_cuda: Optional[bool] = True,
                 device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
                 class_focus: Optional[str] = None,
                 loss_weight: Optional[list[float]] = LOSS_WEIGHT,

                 # openprompt要求参数
                 classes: Optional[list[str]] = None,
                 premodel_name: str = 'bert',
                 premodel_path: str = 'bert-base-cased',
                 special_tokens: Optional[dict] = None,
                 add_special_tokens: Optional[bool] = False,
                 soft_token_num: Optional[int] = 6,
                 template_text_manual: Optional[str] = None,
                 template_text_trained: Optional[str] = None,
                 label_words: Optional[dict] = None
                 ):

        if classes is None:
            classes = CLASS
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS
        if template_text_manual is None:
            template_text_manual = '{"placeholder":"text_a"} It was {"mask"}.'
        if template_text_trained is None:
            template_text_trained = '{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"soft"} {"mask"}.'
        if label_words is None:
            label_words = LABEL_WORDS
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.model_file = model_file
        self.classes = classes
        self.class_focus = class_focus
        self.loss_weight = loss_weight
        self.from_begin = from_begin
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.shuffle = shuffle
        self.max_sentence_length = max_sentence_length
        self.prompt_bias = prompt_bias
        self.premodel_name = premodel_name
        self.premodel_path = premodel_path
        self.special_tokens = special_tokens
        self.add_special_tokens = add_special_tokens
        self.soft_token_num = soft_token_num
        self.use_cuda = use_cuda
        self.device = device

        self.manual_t = True if prompt_bias in ['base', 'trainableVerbalizer'] else False
        self.manual_v = True if prompt_bias in ['base', 'trainableTemplate'] else False
        self.template_text_manual = template_text_manual
        self.template_text_trained = template_text_trained
        self.label_words = label_words
