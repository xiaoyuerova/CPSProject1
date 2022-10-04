from smallSample.Prompt.trainableVerbalizer import main as PromptModel
from recurrent.bertM.test import main as BertModel


def main():
    models = [PromptModel, BertModel]
    for model in models:
        model(
            numbers=['3-t'],
            props=['1p', '2p', '3p', '5p', '7p', '10p', '14p', '20p', '28p', '50p', '70p', '90p', '100p'],
            dir_num=None
        )


if __name__ == '__main__':
    main()
