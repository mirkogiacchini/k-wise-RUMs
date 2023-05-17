import itertools, json, os
from utils import *

def getProbsFromDsiAndDssum(Ds_sum, Dsi, count_threshold):
    slates_inserted = set()
    slates_removed_low_count = 0
    min_frequency = None
    Dsi_out = {}
    for s, freq in Ds_sum.items():
        if freq >= count_threshold:
            if min_frequency is None or freq < min_frequency:
                min_frequency = freq
            slates_inserted.add(s)
            for i in s:
                Dsi_out[(s, i)] = Dsi.get((s, i), 0) / freq
        else:
            slates_removed_low_count += 1
    return Dsi_out, slates_inserted, slates_removed_low_count, min_frequency


def cleanSushi(path, outpath, slate_size, count_threshold = 5):
    Dsi = {}
    Ds_sum = {}

    with open(path, 'r') as f:
        skipFirst = True
        for line in f:
            if skipFirst:
                skipFirst = False
                continue

            s = line.split()
            if len(s) == 0: #!= 12
                continue
            
            s = list(map(int, s[2:]))
            for comb in itertools.combinations(s, slate_size):
                key_comb = tuple(sorted(comb))
                Dsi[(key_comb, comb[0])] = Dsi.get((key_comb, comb[0]), 0) + 1 #comb[0] is the winner
                Ds_sum[key_comb] = Ds_sum.get(key_comb, 0) + 1
    
    Dsi_out, slates_inserted, slates_removed, min_frequency = getProbsFromDsiAndDssum(Ds_sum, Dsi, count_threshold=count_threshold)

    print('slates removed due to low frequency: ', slates_removed)
    print('min frequency in selected slates: ', min_frequency)
    print('number of final slates: ', len(slates_inserted))
    with open(outpath, 'w') as f:
        json.dump({'n':10, 'probs': map_keys_to_json(Dsi_out), 'Dsi_sum': map_keys_to_json(Dsi)}, f)

def cleanSF(path, outpath, slate_size, count_threshold = 5):
    Dsi = {}
    Ds_sum = {}
    n = 0
    original_slate_sizes = {}
    with open(path, 'r') as f:
        skipFirst = True
        for line in f:
            if skipFirst:
                n = len(line.split(','))-1
                skipFirst = False
                continue
            s = list(map(int, line.split(',')))
            winner = s[0] - 1
            big_slate = sorted([i-1 for i in range(1, len(s)) if s[i] == 1])
            if len(big_slate) not in original_slate_sizes:
                original_slate_sizes[len(big_slate)] = set()
            original_slate_sizes[len(big_slate)].add(tuple(big_slate))

            #create all subslates of big_slate having size @slate_size and containing the @winner
            for small_slate in itertools.combinations(big_slate, slate_size-1):
                if winner in small_slate:
                    continue
                slate = tuple(sorted(small_slate + (winner,)))
                Dsi[(slate, winner)] = Dsi.get((slate, winner), 0) + 1
                Ds_sum[slate] = Ds_sum.get(slate, 0) + 1
 
    Dsi_out, slates_inserted, slates_removed_low_count, min_frequency = getProbsFromDsiAndDssum(Ds_sum, Dsi, count_threshold=count_threshold)
    
    print('slates removed due to low frequency: ', slates_removed_low_count)
    print('min frequency in selected slates: ', min_frequency)

    with open(outpath, 'w') as f:
        json.dump({'n':n, 'probs': map_keys_to_json(Dsi_out), 'Dsi_sum': map_keys_to_json(Dsi)}, f)
    
    for k in original_slate_sizes:
        original_slate_sizes[k] = len(original_slate_sizes[k])
    print('original slate sizes: ', original_slate_sizes)
    print('number of final slates: ', len(slates_inserted))
    print('number of slates x winners: ', len(Dsi_out))
    if slate_size is not None:
        assert len(slates_inserted) * slate_size == len(Dsi_out)

def cleanElection(path, outpath, slate_size, count_threshold = 5):
    n = -1
    Dsi = {}  
    Ds_sum = {}
    original_slates = {}
    with open(path, 'r') as f:
        firstLine = True
        for line in f:
            if firstLine:
                firstLine = False
                n = int(line.split()[0])
                continue
            if '"' in line:
                continue
            p = list(map(lambda v: int(v)-1, line.split()[1:-1]))
            if len(p) not in original_slates:
                original_slates[len(p)] = set()
            original_slates[len(p)].add(tuple(sorted(p)))
            
            for comb in itertools.combinations(p, slate_size):
                key_comb = tuple(sorted(comb))
                Dsi[(key_comb, comb[0])] = Dsi.get((key_comb, comb[0]), 0) + 1 #comb[0] is the winner
                Ds_sum[key_comb] = Ds_sum.get(key_comb, 0) + 1

    Dsi_out, slates_inserted, slates_removed, min_freq = getProbsFromDsiAndDssum(Ds_sum, Dsi, count_threshold)
    print('total slates inserted: ', len(slates_inserted))
    print('slates removed due to low frequency: ', slates_removed)
    print('min frequency in inserted slate: ', min_freq)

    with open(outpath, 'w') as f:
        json.dump({'n':n, 'probs': map_keys_to_json(Dsi_out), 'Dsi_sum': map_keys_to_json(Dsi)}, f)

    for k in original_slates:
        original_slates[k] = len(original_slates[k])
    print('original slate sizes: ', original_slates)
    assert len(slates_inserted) * slate_size == len(Dsi_out)


if __name__ == '__main__':
    os.makedirs('./data/clean/election', exist_ok=True)

    slate_sizes = range(2, 6)
    for i in slate_sizes:
        cleanSushi('./data/raw/sushi3-2016/sushi3a.5000.10.order', f'./data/clean/sushiA_{i}slates', i)

        cleanSF('./data/raw/SFshop.csv', f'./data/clean/SFshop_{i}slates', i) 
        cleanSF('./data/raw/SFwork.csv', f'./data/clean/SFwork_{i}slates', i)  

        cleanElection('./data/raw/a5.hil', f'./data/clean/election/a5_{i}slates', i)
        cleanElection('./data/raw/a9.hil', f'./data/clean/election/a9_{i}slates', i)
        cleanElection('./data/raw/a17.hil', f'./data/clean/election/a17_{i}slates', i)
        cleanElection('./data/raw/a48.hil', f'./data/clean/election/a48_{i}slates', i)
        cleanElection('./data/raw/a81.hil', f'./data/clean/election/a81_{i}slates', i)