import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Utils import *

if __name__ == '__main__':
    # x = np.array([1, 8, 10])
    # y = np.array([3, 10, 5])
    # file = __file__
    # plt.title('os.path.basename(file)')
    # plt.plot(x, y)
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.show()
    # path = os.path.join(os.path.dirname(__file__), './outputs/{}'.format('123'))
    # print(os.path.dirname(file))
    # plt.savefig(path)
    cr = json.load(open('outputs/recurrent-bertM-test/test3-2.json'))
    print(cr[0])

