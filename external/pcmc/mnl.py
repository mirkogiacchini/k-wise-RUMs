'''
this implementation is from https://github.com/sragain/pcmc-nips
'''

import numpy as np
from scipy.optimize import minimize
import random

def solve_ctmc(Q):
	"""Solves the stationary distribution of the CTMC whose rate matrix matches
	the input on off-diagonal entries. 
	Arguments:
	Q- rate matrix
	"""
	A=np.copy(Q)
	for i in range(Q.shape[0]):
		A[i,i] = -np.sum(Q[i,:])
	n=Q.shape[0]
	A[:,-1]=np.ones(n)
	b= np.zeros(n)
	b[n-1] = 1
	if np.linalg.matrix_rank(A)<Q.shape[0]:
		print(Q)
		print(A)
		assert(False)
	return np.linalg.solve(A.T,b)


def ILSR(C,n):
	"""performs the ILSR algorithm to learn optimal MNL weights for choice data
	
	Arguments:
	C- dictionary containing choice data
	n- number of elements in the union of the choice sets
	epsilon- hyperparameter used for termination
	"""
	pi = np.ones(n).astype(float)/n
	diff = 1
	epsilon = 10**(-6)	
	while diff>epsilon:
		pi_ = pi
		lam = np.ones((n,n))*epsilon #initialization>0 prevents numerical issue
		for S in C:
			gamma = np.sum([pi[x] for x in S])
			pairs = [(i,j) for i in range(len(S)) for j in range(len(S)) if j!=i]
			for i,j in pairs:
				lam[S[j],S[i]]+=C[S][i]/gamma
			
		pi = solve_ctmc(lam)
		diff = np.linalg.norm(pi_-pi)
	return pi