from dIVA_L import dIVA_L
import numpy as np
from scipy.io import loadmat

from test1 import test1
from test2 import test2
from test3 import test3

def main() :
    S = np.zeros((16,3296,3))
    X = S.copy()
    Xm = []
    A = np.random.rand(16,16,3)
    for i in range(3) :
        S = loadmat("SCV_IVA_case12_r001.mat")['Sgt'][0,i][:,0:3296]
        X[:,:,i] = np.dot(A[:,:,i], S[:,:])
    test1(X=X, A=A, verbose=True, subj_per_site=2)
    test2(X=X, A=A, verbose=True, num_sites=2)
    test3(X=X, A=A, verbose=True)

main()
