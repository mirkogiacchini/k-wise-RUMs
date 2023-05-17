import sys
from predictions import get_statistics

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

    #recap evaluation
    for ds_nm in datasets:
        for slate_size in slate_sizes:
            print(f'==== {ds_nm} --- {slate_size}')
            print(f'train tensor, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_trainmatrix'))
            print(f'rumrunner, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_rumrunner'))
            print(f'MNL, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_MNL'))
            print(f'MNL pcmc, {ds_nm} {slate_size}-slates: ', get_statistics(f'./predictions/{ds_nm}_{slate_size}slates_MNL_pcmc'), '\n')