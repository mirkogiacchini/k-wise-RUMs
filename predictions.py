import rumrunner
from rumrunner import Dataset
import random, time, json, os, math, sys
from evaluator import slate_probabilities_from_rum
from cleaner import getProbsFromDsiAndDssum
from MNL import *
from pcmpMNL import train_MNLpcmc, pred_from_MNLpcmc
from utils import *
import numpy as np

def RMSE(pred_Dsi, real_Dsi, ssize):
    error = 0
    assert pred_Dsi.keys() == real_Dsi.keys()
    for key in real_Dsi.keys():
        error += (real_Dsi[key] - pred_Dsi[key]) * (real_Dsi[key] - pred_Dsi[key])
    return math.sqrt(error / ssize)

def get_probs_from_list(l):
    Dsi = {}
    Ds_sum = {}
    for v in l:
        slate = tuple(v[0])
        winner = v[1]
        Dsi[(slate, winner)] = Dsi.get((slate, winner), 0) + 1
        Ds_sum[slate] = Ds_sum.get(slate, 0) + 1
    return getProbsFromDsiAndDssum(Ds_sum, Dsi, 0)[0]

def get_sums_from_list(l):
    Dsi = {}
    for v in l:
        slate = tuple(v[0])
        winner = v[1]
        Dsi[(slate, winner)] = Dsi.get((slate, winner), 0) + 1
    return Dsi

def get_statistics(dir):
    error_list = []
    for f in os.listdir(dir):
        if f.split('_')[-1] == 'result':
            with open(dir+'/'+f, 'r') as ff:
                error = json.load(ff)['error']
                error_list += [error]
    errornp = np.array(error_list)
    return np.mean(errornp), np.std(errornp)

def predict_via_training_data(train_probs, test_slates, slate_size):
    preds = {}
    for slate in test_slates:
        for i in slate:
            preds[(slate, i)] = train_probs.get((slate, i), 1/slate_size) #uniform probability for out-of-dataset slates
    return preds

def kfoldCrossVal(n, slate_list, k=5, seeds = [42, 43], out_folder = '.', slate_size = None, model='rumrunner'):
    os.makedirs(out_folder, exist_ok = True)
    start_time = time.time()
    
    total_iterations = len(seeds) * k
    it = 0

    all_errors = []

    for seed in seeds:
        rnd = random.Random(seed)
        rnd.shuffle(slate_list)
        for iter in range(k):
            it += 1
            print(f'seed: {seed} - iter: {iter} --- {it}/{total_iterations}')
            tt = time.time()
            train_list = []
            test_list = []

            for i in range(len(slate_list)):
                if i%k == iter: #test                        
                    test_list.append(slate_list[i])
                else: #train
                    train_list.append(slate_list[i])
            
            test_probs = get_probs_from_list(test_list)
            S_test = set()
            for key in test_probs:
                S_test.add(key[0])

            if model in ['rumrunner', 'trainmatrix']:
                train_probs = get_probs_from_list(train_list)

            if model == 'rumrunner':
                S_train = set()
                for key in train_probs:
                    S_train.add(key[0])

                rum = rumrunner.rumrunner_(n, train_probs, S_train, seed=seed, load_checkpoint=None, \
                    checkpoint_path=f'{out_folder}/seed-{seed}_iter-{iter}_rumrunner', epsilon_D_tolerance=0.00001, convergence_count=20, \
                    epsilon_convergence=0.0001, slate_size=slate_size, max_iterations=250, verbose=False, checkpoint_iter=100)
                
                test_preds = slate_probabilities_from_rum(S_test, rum)
            elif model == 'MNL':
                mnl = train_MNL(n, train_list)
                test_preds = pred_from_MNL(mnl, S_test, n)
            elif model == 'MNLpcmc':
                mnl = train_MNLpcmc(n, get_sums_from_list(train_list))
                test_preds = pred_from_MNLpcmc(mnl, S_test, n)
            elif model == 'trainmatrix':
                test_preds = predict_via_training_data(train_probs, S_test, slate_size)
            else:
                assert False
            
            #error = avg_l1_error(len(S_test), test_probs, test_preds)
            error = RMSE(test_preds, test_probs, len(S_test))
            all_errors.append(error)
            print('iteration error: ', error)
            print(f'iteration took {(time.time() - tt):.2f}s')
            with open(f'{out_folder}/seed-{seed}_iter-{iter}_result', 'w') as f:
                json.dump({'error': error, 'time': time.time()-start_time, 'model': model}, f)

    stdev = np.std(np.array(all_errors))
    avg = np.mean(np.array(all_errors))
    print('average error: ', avg)
    print('stddev: ', stdev)
    print('total time: ', time.time() - start_time)

#old mse, only for 2-slates
#def rmse_2slates(D1, D2, n):
#    keys = ((i,j) for i in range(n) for j in range(i+1, n))
#    assert D1.keys() == D2.keys()
#    toterr = 0
#    for k in keys:
#        if (k, k[1]) in D1: toterr += (D1[(k, k[1])] - D2[(k, k[1])])**2
#    return toterr**0.5 / n


if __name__ == '__main__':
    if len(sys.argv) == 1:
        datasets = ['sushiA', 'SFwork', 'SFshop', 'election/a9', 'election/a17', 'election/a48', 'election/a81']
        slate_sizes = [2, 3, 4, 5]
    elif len(sys.argv) == 3:
        datasets = [sys.argv[1]]
        slate_sizes = [int(sys.argv[2])]
    else:
        print('python predictions.py [dataset_name] [slate_size]')
        assert False

    os.makedirs('./predictions/election', exist_ok=True)
    for slate_size in slate_sizes:
        for ds_nm in datasets:
            print(ds_nm, slate_size)
            ds = Dataset(f'./data/clean/{ds_nm}_{slate_size}slates', keep_sum=True)
            #kfoldCrossVal(ds.n, list_from_sums(ds.SUM_DSI), 5, seeds=[42+i for i in range(10)], out_folder=f'./predictions/{ds_nm}_{slate_size}slates_rumrunner', slate_size=slate_size, model='rumrunner')
            #kfoldCrossVal(ds.n, list_from_sums(ds.SUM_DSI), 5, seeds=[42+i for i in range(10)], out_folder=f'./predictions/{ds_nm}_{slate_size}slates_MNL', model='MNL')
            kfoldCrossVal(ds.n, list_from_sums(ds.SUM_DSI), 5, seeds=[42+i for i in range(10)], out_folder=f'./predictions/{ds_nm}_{slate_size}slates_MNL_pcmc', model='MNLpcmc')
            #kfoldCrossVal(ds.n, list_from_sums(ds.SUM_DSI), 5, seeds=[42+i for i in range(10)], out_folder=f'./predictions/{ds_nm}_{slate_size}slates_trainmatrix', slate_size=slate_size, model='trainmatrix')
    
    #recap evaluation
    for slate_size in slate_sizes:
        for ds_nm in datasets:
            #print(f'==== {ds_nm} --- {slate_size}')
            #print(f'train tensor, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_trainmatrix'))
            #print(f'rumrunner, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_rumrunner'))
            #print(f'MNL, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_MNL'), '\n')
            print(f'MNL pcmc, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_MNL_pcmc'), '\n')