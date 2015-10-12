from dIVA_L import dIVA_L
import numpy as np
from scipy.io import loadmat

from test1 import test1
from test2 import test2
from test4 import test4

def main () :
    S = loadmat("d_variables.mat")
    N,R,K = S.shape
    X = S.copy()
    A = np.random.rand(N,N,K)
    for i in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    test1(X=X, A=A, verbose=True, subj_per_site=2)
    test2(X=X, A=A, verbose=True, num_sites=2)
    test4(X=X, A=A, verbose=True)
