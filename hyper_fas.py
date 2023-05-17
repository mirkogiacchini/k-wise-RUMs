import itertools, random

def hyper_fas_cost(perm, delta, slate_size):
    return sum(delta.get((tuple(sorted(c)), c[-1]), 0) for c in itertools.combinations(perm, slate_size))

def random_localsearch_hyper_fas(seed, runs, n, delta, eps_improv, D, eps_less_D, slate_size, num_iter, min_number_iter = 0):
    m = None
    rnd = random.Random(seed+num_iter) #A5, 3 slates: +1
    p = list(range(n))
    for r in range(runs):
        rnd.shuffle(p) #sample random permutation
        cost_p = hyper_fas_cost(p, delta, slate_size)
        best_p = p
        it = 0
        while True:
            in_cp = cost_p
            it += 1

            for i in range(n): #take the "best" single movement
                for j in range(n):
                    if i != j:
                        if i < j:
                            q = p[:i] + p[i+1:j+1] + [p[i]] + p[j+1:] 
                        else:
                            q = p[:j] + [p[i]] + p[j:i] + p[i+1:]
                        c = hyper_fas_cost(q, delta, slate_size)
                        if c < cost_p:
                            cost_p = c
                            best_p = q

            p = best_p
            if cost_p >= in_cp - eps_improv: #improvement smaller than epsilon, stop
                break
        
        if m == None or cost_p < m[0]:
            m = (cost_p, best_p[:])
        if m[0] < D - eps_less_D and r+1 >= min_number_iter:
            break
    return m

def dynamic_programming_hypter_fas(n, delta, slate_size):
    N = set(range(n))

    sol = {tuple(): (0, tuple())}
    for c in range(n):
        tmp_sol = {}
        for s in itertools.combinations(range(n), c):
            R = sorted(N - set(s))
            #choose how to augment "s"
            for i in R: #augment with R
                v = sol[s][0]
                if len(s) >= slate_size-1:
                    for j in itertools.combinations(s, slate_size - 1): #O(n^{k-1})
                        tp = tuple(sorted(j + (i,)))
                        if (tp, i) in delta:
                            v += delta[(tp, i)]
                x = tuple(sorted(s + (i,)))
                if x not in tmp_sol or v < tmp_sol[x][0]:
                    tmp_sol[x] = (v, sol[s][1] + (i,))
        sol = tmp_sol

    return sol[tuple(range(n))]

def brute_force_hyper_fas(n, delta, old_perms, slate_size):
    old_perms_s = set([tuple(p) for p in old_perms])
    mn = None
    mnPerm = None
    for p in itertools.permutations(range(n)):
        if tuple(p) in old_perms_s:
            continue
        c = hyper_fas_cost(p, delta, slate_size)
        if mn is None or c < mn:
            mn = c
            mnPerm = p
    return mn, mnPerm