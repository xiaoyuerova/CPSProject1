import torch


class Parameters:
    # 句子最大长度
    max_length = 16

    # 喂数据的batch size
    Batch_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_names = ['SESU', 'SMC', 'SSI', 'SN', 'CRF', 'CP', 'CEC', 'CMC']

    epochs = 15

    lr = 2e-3

    log_interval = 50


parameters = Parameters()
