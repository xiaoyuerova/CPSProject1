from smallSample.Prompt.trainableVerbalizer import main as PromptModel
from smallSample.KNN.main import main as KNN
from smallSample.Linear.main import main as Linear
from smallSample.RF.RF import main as RF
from smallSample.textCNN.main import main as textCNN
from smallSample.textGRU.main import main as textGRU
from smallSample.textLSTM.main import main as textLSTM
from recurrent.bertM.test import main as BertModel


def main():
    models = [PromptModel, KNN, Linear, RF, textCNN, textGRU, textLSTM, BertModel]
    for model in models:
        model(
            numbers=['1', '2', '3', '4', '5'],
            props=['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p'],
            dir_num=None
        )


if __name__ == '__main__':
    print('ok')
    main()
