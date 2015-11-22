import numpy as np

from scipy.io import loadmat
from measure_indep import measure_indep
import sys

if __name__ == "__main__" : 
    sys.argv.pop(0)
    KK = len(sys.argv)
    K = 10
    N,R = loadmat(sys.argv[0])['Sgt'][0,0].shape
    S = np.zeros((N,R,K))
    for kk in range(KK) :
        SS = loadmat(sys.argv[kk])['Sgt']
        for k in range(K) :
            S[:,:,k] = SS[0,k]
        measure_indep(S, "SCV_%i"%kk)
