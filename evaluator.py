from rumrunner import Dataset
import hyper_fas as fas
from utils import *
import json, os
import matplotlib.pyplot as plt
import sys

def slate_probabilities_from_rum(S, perm_probs):
    ds = {}
    for slate in S:
        for i in slate:
            ds[(tuple(slate), i)] = 0
            for p, prob in perm_probs.items():
                if p.index(i) == max([p.index(j) for j in slate]):
                    ds[(tuple(slate), i)] += prob  
    return ds

def reduce_permutations(perm_prob):
    ret = {}
    for perm,prob in perm_prob.items():
        if prob > 0:
            ret[perm] = prob
    return ret

def avg_l1_error(Ssize, p1, p2):
    error = 0
    assert p1.keys() == p2.keys()
    for p,prob in p2.items():
        prob1 = p1.get(p, 0)
        error += abs(prob - prob1)
    return error / Ssize

def inverse_perm_prob_json(l):
    return {tuple(d['key']):d['value'] for d in l}

def error_statistics(d1, d2, plot_prefix):
    error_per_slate = {}
    assert d1.keys() == d2.keys()
    for p,prob in d1.items():
        error_per_slate[p[0]] = error_per_slate.get(p[0], 0) + abs(prob - d2[p])
    errors = sorted([v for k,v in error_per_slate.items()])
    
    if plot_prefix is not None:
        perc = [(i+1)/len(errors) for i in range(len(errors))]
        plt.figure()
        plt.xlabel('error')
        plt.ylabel('%slates')
        #plt.scatter(errors, perc)
        plt.plot(errors, perc, drawstyle='steps-post')
        plt.vlines(sum(errors) / len(errors), 0, 1, colors=['r'], linestyles='dashed', label='mean')
        plt.legend()
        plt.savefig(plot_prefix+'_error_distribution.pdf')
        #plt.show()

def get_lower_bound(n, S, delta, D, lp_value, slate_size):
    if slate_size is None:
        min_cost, _ = fas.brute_force_hyper_fas(n, S, delta, [])
    else:
        min_cost, _ = fas.dynamic_programming_hypter_fas(n, delta, slate_size)

    #print(min_cost, p, fas.hyper_fas_cost(p, S, delta), min_cost2, p2, fas.hyper_fas_cost(p2, S, delta))
    if min_cost >= D:
        return lp_value
    x = D - min_cost
    print('D: ', D, '  min_cost: ', min_cost, '  diff: ', x)
    return lp_value - x

def evaluate(dataset, checkpoint, plot_prefix = None, computeLB = False, slate_size = None):
    '''
    check error between a dataset and a rum (checkpointed) on the same dataset
    '''
    with open(checkpoint, 'r') as f:
        d = json.load(f)
        probs = inverse_perm_prob_json(d['perms_probability'])
        if computeLB:
            delta = inverse_map_json(d['delta'])
            D = d['D']
        lp_value = d['value']
        previous_size = len(probs)
        probs = reduce_permutations(probs)
        slate_probs = slate_probabilities_from_rum(dataset.S, probs)
    
    print('Support size: ', len(probs))
    print('removed permutations: ', previous_size - len(probs))
    print('num iterations: ', previous_size)

    #compare dataset.Dsi with slate_probs
    avgl1 = avg_l1_error(len(dataset.S), dataset.Dsi, slate_probs)
    print('avg l1 error: ', avgl1)
    assert abs(avgl1 - lp_value) < 0.001, 'the errors do not match, check precision issues in rumrunner'

    error_statistics(dataset.Dsi, slate_probs, plot_prefix)

    if computeLB:
        lb = get_lower_bound(dataset.n, dataset.S, delta, D, lp_value, slate_size)
        print('lower bound: ', lb)

def plot_many_error_statistics(dataset_names, checkpoint_names, names, rows, cols, out='./plots/statplot.pdf'):
    assert rows * cols == len(dataset_names) == len(checkpoint_names) == len(names)

    fig, axs = plt.subplots(nrows=rows, ncols=cols)
    fig.supxlabel('error')
    fig.supylabel('fraction of slates')

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            ds = Dataset(dataset_names[idx])
            with open(checkpoint_names[idx], 'r') as f:
                d = json.load(f)
                probs = inverse_perm_prob_json(d['perms_probability'])
                probs = reduce_permutations(probs)
                probs = slate_probabilities_from_rum(ds.S, probs)
            error_per_slate = {}

            for p,prob in ds.Dsi.items():
                error_per_slate[p[0]] = error_per_slate.get(p[0], 0) + abs(prob - probs[p])
            errors = sorted([v for k,v in error_per_slate.items()])
            perc = [(i+1)/len(errors) for i in range(len(errors))]
            axs[r, c].plot(errors, perc, drawstyle='steps-post')
            axs[r, c].vlines(sum(errors) / len(errors), 0, 1, colors=['r'], linestyles='dashed', label='mean error' if idx == 0 else None)
            axs[r, c].set_title(names[idx])

    fig.legend(loc='lower left')
    fig.tight_layout() 
    plt.savefig(out)
    plt.show()

if __name__ == '__main__':

    if len(sys.argv) == 1:
        datasets = ['sushiA', 'SFwork', 'SFshop', 'election/a5', 'election/a9', 'election/a17', 'election/a48', 'election/a81']
        slate_sizes = [2, 3, 4, 5]
    elif len(sys.argv) == 3:
        datasets = [sys.argv[1]]
        slate_sizes = [int(sys.argv[2])]
    else:
        print('python evaluator.py [dataset_name] [slate_size]')
        assert False
    
    compute_lb = [True] * len(datasets)
    os.makedirs('./plots/election', exist_ok=True)
    for slate_size in slate_sizes:
        for i,ds_nm in enumerate(datasets):
            ds = Dataset(f'./data/clean/{ds_nm}_{slate_size}slates')
            checkpoint = f'./checkpoint/{ds_nm}_{slate_size}slates_ckpt'
            print(f'\n===== {ds_nm} {slate_size}slates')
            evaluate(ds, checkpoint, plot_prefix=f'./plots/{ds_nm}_{slate_size}slates', slate_size=slate_size, computeLB=compute_lb[i])
    

    #Plot from paper
    #ds_names = ['./data/clean/election/a9_4slates', 
    #    './data/clean/election/a17_4slates',
    #    './data/clean/election/a48_4slates',
    #    './data/clean/election/a81_4slates',
    #    './data/clean/SFwork_4slates',
    #    './data/clean/SFshop_4slates'
    #]
    #checkpoint_names = ['./checkpoint/election/a9_4slates_ckpt', 
    #    './checkpoint/election/a17_4slates_ckpt',
    #    './checkpoint/election/a48_4slates_ckpt',
    #    './checkpoint/election/a81_4slates_ckpt',
    #    './checkpoint/SFwork_4slates_ckpt',
    #    './checkpoint/SFshop_4slates_ckpt'
    #]
    #names =['A9', 'A17', 'A48', 'A81', 'SFwork', 'SFshop']
    #plot_many_error_statistics(ds_names, checkpoint_names, names, 2, 3, out='./plots/statplot4.pdf')
