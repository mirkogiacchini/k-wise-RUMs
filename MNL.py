from sklearn.linear_model import LogisticRegression
import numpy as np
from rumrunner import Dataset
from evaluator import avg_l1_error
from utils import *
import sys

def train_MNL(n, train_list, penalty='l2'):
    lr = LogisticRegression(penalty=penalty, multi_class='ovr', max_iter=100) #increasing iterations does not help
    X = []
    y = []
    for l in train_list:
        one_hot = [0] * n
        for v in l[0]:
            one_hot[v] = 1
        X.append(one_hot)
        y.append(l[1])

    X = np.array(X)
    y = np.array(y)
    print('X, y shapes: ', X.shape, y.shape)
    lr.fit(X, y)
    print('lr score: ', lr.score(X, y))
    return lr

def pred_from_MNL(mnl, S_test, n):
    X_test = []
    l_test = list(S_test)
    for slate in l_test:
        one_hot = [0] * n
        for v in slate:
            one_hot[v] = 1
        X_test.append(one_hot)
    X_test = np.array(X_test)
    probs = mnl.predict_proba(X_test)

    Dsi = {}
    for i, x in enumerate(l_test):
        sm = sum([probs[i][j] for j in x]) #only element in x have non-zero probability
        for j in x:
            Dsi[(tuple(x), j)] = probs[i][j] / sm

    return Dsi

def evaluateMNL(dataset):
    lr = train_MNL(dataset.n, list_from_sums(dataset.SUM_DSI), penalty='none') #we want to overfit the training set, remove regularization
    preds = pred_from_MNL(lr, dataset.S, dataset.n)
    return avg_l1_error(len(dataset.S), preds, dataset.Dsi)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        datasets = ['sushiA', 'SFwork', 'SFshop', 'election/a5', 'election/a9', 'election/a17', 'election/a48', 'election/a81']
        slate_sizes = [2, 3, 4, 5]
    elif len(sys.argv) == 3:
        datasets = [sys.argv[1]]
        slate_sizes = [int(sys.argv[2])]
    else:
        print('python MNL.py [dataset_name] [slate_size]')
        assert False

    for slate_size in slate_sizes:
        for ds_nm in datasets:
            ds = Dataset(f'./data/clean/{ds_nm}_{slate_size}slates', keep_sum=True)
            print(f'{ds_nm} {slate_size} slates (MNL): {evaluateMNL(ds)} \n\n')