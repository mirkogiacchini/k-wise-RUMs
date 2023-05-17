import json
from docplex.mp.model import Model
from utils import *
import hyper_fas as fas
import time, os, sys

class Dataset():
    def __init__(self, path, keep_sum = False):
        self.name = path
        with open(path, 'r') as f:
            data = json.load(f)
            self.n = data['n']
            self.Dsi = inverse_map_json(data['probs'])
            if keep_sum:
                self.SUM_DSI = inverse_map_json(data['Dsi_sum'])
        self.S = set()
        for key in self.Dsi:
            self.S.add(key[0])    

        for s in self.S:
            for i in s:
                assert (s, i) in self.Dsi

class LinearProgram():
    def __init__(self, Dsi, S, slate_size):
        self.m = Model()
        self.slate_size = slate_size
        self.m.parameters.lpmethod = 4 
        self.m.parameters.parallel = -1
        
        self.D = self.m.continuous_var(lb=-self.m.infinity)

        self.Si = list(Dsi.keys())
        self.delta = self.m.continuous_var_dict(keys=self.Si, lb=-self.m.infinity)

        self.m.maximize(self.D-self.m.scal_prod((self.delta[si] for si in self.Si), (Dsi[si] for si in self.Si)))

        normalizer = len(S) #n^k roughly
        self.m.add_constraints(normalizer*self.delta[si] <= 1 for si in self.Si)
        self.m.add_constraints(normalizer*self.delta[si] >= -1 for si in self.Si)
        self.perms = {}

        self.oldVal = None

    def _permutation_constraint(self, p):
        inv_p = [0] * len(p)
        for i in range(len(p)):
            inv_p[p[i]] = i
        return self.m.sum(self.delta[si] for si in self.Si if inv_p[si[1]] == max([inv_p[j] for j in si[0]])) >= self.D
    
    def add_p(self, p): 
        self.perms[tuple(p)] = self.m.add_constraint(self._permutation_constraint(p))
    
    def solve(self): # D, delta, solution_value, Perms
        sol = self.m.solve(clean_before_solve=False) #do not clean to start from previous solutions
        assert sol is not None, f'instance not solvable: {self.m.solve_details}'
        
        self.oldVal = {'D': sol[self.D], 'delta': {si: sol[self.delta[si]] for si in self.Si}}

        return self.oldVal['D'], self.oldVal['delta'], sol.objective_value, {p:-v for p,v in zip(self.perms.keys(), self.m.dual_values(self.perms.values()))}

def rumrunner_(n, Dsi, S, seed=42, load_checkpoint = None, checkpoint_path = None, checkpoint_iter = 10, epsilon_D_tolerance=0.01, min_ls_iter = 5,
                convergence_count = 20, epsilon_convergence = 0.0001, slate_size = None, max_iterations = None, verbose = True): 
    start_time = time.time()
    model = LinearProgram(Dsi, S, slate_size)

    iter = 0
    if load_checkpoint is not None:
        with open(load_checkpoint, 'r') as f:
            d = json.load(f)
            if verbose: print('resumed value: ', d['value'])
            iter = d['num_iter']
            perms = d['perms']
            if 'time_passed' in d:
                start_time -= d['time_passed']
    else:
        perms = [list(range(n))]

    for p in perms:
        model.add_p(p)
    
    errs = []

    best_lower_bound = 0
    tt = time.time()
    while True:
        iter += 1
        if verbose: print(iter)
        tt = time.time()
        D, delta, value, perms_p = model.solve()
        if verbose: print(f'LP solved in {(time.time() - tt):.2f}s')
        errs += [value]
        if len(errs) > convergence_count:
            errs = errs[1:]
            #assert len(errs) == convergence_count
        
        if verbose:
            print('D: ', D)
            print('value: ', value)

        tt = time.time()
        exact_min = True
        if n <= 13:
            new_p = fas.dynamic_programming_hypter_fas(n, delta, slate_size)
        else: #(min_perm_value, perm)
            new_p = fas.random_localsearch_hyper_fas(seed, 100, n, delta, 0.001, D, epsilon_D_tolerance, slate_size, iter, min_number_iter=min_ls_iter)
            exact_min = False

        if verbose:
            print(f'hyperplane search took {(time.time() - tt):.2f}s')
            print('min_cost: ', new_p[0], '  D - min_cost: ', D - new_p[0])
        if exact_min:
            best_lower_bound = max(best_lower_bound, value - D + new_p[0])
            if verbose: print('best lower bound: ', best_lower_bound)

        #check exit conditions
        if new_p[0] >= D or (len(errs) == convergence_count and errs[0] - errs[-1] < epsilon_convergence) or \
            (max_iterations is not None and iter >= max_iterations) or abs(best_lower_bound - value) < 0.0000000001: 
            if verbose:
                if new_p[0] >= D: #all constraints satisfied (modulo local search mistakes)
                    print('smallest error >= D')
                elif abs(best_lower_bound - value) < 0.0000000001:
                    print('optimal value reached')
                elif max_iterations is not None and iter >= max_iterations:
                    print('max iterations reached')
                else:
                    print('convergence reached')
            break
        
        #assert tuple(new_p[1]) not in perms
        model.add_p(new_p[1])
        perms.append(tuple(new_p[1]))

        if checkpoint_path is not None and iter % checkpoint_iter == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump({'perms': perms, 'value': value, 'num_iter': iter, 'perms_probability': map_keys_to_json(perms_p),
                            'D': D, 'delta': map_keys_to_json(delta), 'time_passed': time.time() - start_time, 'bestLB': best_lower_bound}, f)

    if checkpoint_path is not None:
        with open(checkpoint_path, 'w') as f:
            json.dump({'perms': perms, 'value': value, 'num_iter': iter, 'perms_probability': map_keys_to_json(perms_p),
                        'D': D, 'delta': map_keys_to_json(delta), 'time_passed': time.time() - start_time, 'bestLB': best_lower_bound}, f)
    return perms_p

def rumrunner(dataset, seed=42, load_checkpoint = None, checkpoint_path = None, checkpoint_iter = 10, epsilon_D_tolerance=0.01, min_ls_iter=0,
                convergence_count = 20, epsilon_convergence = 0.0001, slate_size = None, max_iterations = None, verbose = True): 
    return rumrunner_(dataset.n, dataset.Dsi, dataset.S, seed=seed, load_checkpoint=load_checkpoint, checkpoint_path=checkpoint_path, checkpoint_iter=checkpoint_iter, 
                    epsilon_D_tolerance=epsilon_D_tolerance, convergence_count=convergence_count, epsilon_convergence=epsilon_convergence,  
                    slate_size=slate_size, max_iterations=max_iterations, verbose=verbose, min_ls_iter=min_ls_iter)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        datasets = ['sushiA', 'SFwork', 'SFshop', 'election/a5', 'election/a9', 'election/a17', 'election/a48', 'election/a81']
        slate_sizes = [2, 3, 4, 5]
    elif len(sys.argv) == 3:
        datasets = [sys.argv[1]]
        slate_sizes = [int(sys.argv[2])]
    else:
        print('python rumrunner.py [dataset_name] [slate_size]')
        assert False
    
    os.makedirs('./checkpoint/election', exist_ok=True)
    for slate_size in slate_sizes:
        for ds_nm in datasets:
            print(ds_nm, slate_size)
            ds = Dataset(f'./data/clean/{ds_nm}_{slate_size}slates')
            min_ls_iter = 5 if slate_size < 5 else 1
            max_iter = 1500
            if slate_size == 4:
                if ds_nm == 'election/a17':
                    max_iter = 1580
                elif ds_nm == 'election/a48':
                    max_iter = 1564
            rumrunner(ds, load_checkpoint=None, checkpoint_path=f'./checkpoint/{ds_nm}_{slate_size}slates_ckpt', convergence_count=20, epsilon_convergence=0.00001, epsilon_D_tolerance=0.001, slate_size=slate_size, max_iterations=max_iter, min_ls_iter=min_ls_iter)
        