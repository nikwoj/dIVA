from dIVA_L import dIVA_L
import numpy as np
from scipy.io import loadmat

from test1 import test1
from test2 import test2
from test3 import test3
from test4 import test4

def main() :
    #N,R = loadmat("SCV_IVA_case12_r001.mat")['Sgt'][0,0].shape # In case of SCV_IVA_case... N=16, R=32968
    N,R = 16, 3000
    K = 10 # Also, K=10 for SCV_IVA_case...
    X = np.zeros((N,R,K))
    Xm = []
    A = np.random.rand(N,N,K)
    for i in range(K) :
        S = loadmat("SCV_IVA_case12_r001.mat")['Sgt'][0,i][:,0:R]
        X[:,:,i] = np.dot(A[:,:,i], S[:,:])
    #test1(X=X, A=A, verbose=True, subj_per_site=2)
    #test2(X=X, A=A, verbose=True, num_sites=2)
    #test3(X=X, A=A, verbose=True)
    test4(X=X, A=A, verbose=True)
main()
