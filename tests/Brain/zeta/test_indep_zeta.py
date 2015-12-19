import numpy as np
from  measure_indep import measure_indep
from scipy.io import loadmat
import sys


if __name__ == "__main__" :
    sys.argv.pop(0)
    K = len(sys.argv)
    #K = 2
    print K
    N,R = loadmat(sys.argv[0])['SM'].shape
    S = np.zeros((N,R,K))
    for k in range(K) :
        S[:,:,k] = loadmat(sys.argv[k])['SM'][:,:]
    measure_indep(S, "zeta")
