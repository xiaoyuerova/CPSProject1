import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

# 准备数据集
dir_train = os.path.join(os.path.dirname(__file__), '../../data/single-sentence-prediction/train_data.csv')
dir_test = os.path.join(os.path.dirname(__file__), '../../data/single-sentence-prediction/test_data.csv')
train_data = pd.read_csv(dir_train)
test_data = pd.read_csv(dir_test)
tfidf = TfidfVectorizer(norm='l2', max_features=10000, use_idf=True, smooth_idf=True, sublinear_tf=False,
                        ngram_range=(1, 3))
tfidf.fit_transform(train_data['Text'])
train_x = tfidf.transform(train_data['Text'])
test_x = tfidf.transform(test_data['Text'])
train_y = train_data['Label'].to_numpy()
test_y = test_data['Label'].to_numpy()
clf = RandomForestClassifier()
clf.fit(train_x, train_y)
pre = clf.predict(test_x)
print(accuracy_score(test_y, pre))
print('test kappa: ', cohen_kappa_score(test_y, pre))
print(classification_report(test_y, pre))
