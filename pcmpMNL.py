from external.pcmc.mnl import ILSR
import numpy as np
from rumrunner import Dataset
from evaluator import avg_l1_error
import sys

def train_MNLpcmc(n, Dsi_sum):
    dsi_index = {}
    for (S, w) in Dsi_sum:
        if S not in dsi_index:
            dsi_index[S]=np.ones(len(S)) * (1e-5) #avoids precision issues
        dsi_index[S][S.index(w)] = Dsi_sum[(S, w)]
    
    return ILSR(dsi_index, n)

def pcmcPredict(weights, slate):
    norm = 0
    for x in slate:
        norm += weights[x]
    preds = {}
    for x in slate:
        preds[x] = weights[x] / norm
    return preds

def pred_from_MNLpcmc(mnl, S_test, n):
    preds = {}
    for slate in S_test:
        pslate = pcmcPredict(mnl, slate)
        for x in pslate:
            preds[(slate, x)] = pslate[x]
    return preds

def evaluateMNLpcmc(dataset):
    lr = train_MNLpcmc(dataset.n, dataset.SUM_DSI) 
    preds = pred_from_MNLpcmc(lr, dataset.S, dataset.n)
    return avg_l1_error(len(dataset.S), preds, dataset.Dsi)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        datasets = ['sushiA', 'SFwork', 'SFshop', 'election/a5', 'election/a9', 'election/a17', 'election/a48', 'election/a81']
        slate_sizes = [2, 3, 4, 5]
    elif len(sys.argv) == 3:
        datasets = [sys.argv[1]]
        slate_sizes = [int(sys.argv[2])]
    else:
        print('python3 pcmpMNL.py [dataset_name] [slate_size]')
        assert False

    for ds_nm in datasets:
        for slate_size in slate_sizes:
            ds = Dataset(f'./data/clean/{ds_nm}_{slate_size}slates', keep_sum=True)
            print(f'{ds_nm} {slate_size} slates (MNL pcmc): {evaluateMNLpcmc(ds)} \n')