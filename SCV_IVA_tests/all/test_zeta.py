import numpy as np

from IVA_L import iva_l
from condition_matrix import fix_cond
from scipy.io import loadmat
from test4 import test4
import sys




if __name__ == "__main__" :
    sys.argv.pop(0)
    K = len(sys.argv)
    N,R = loadmat(sys.argv[0])['SM'].shape
    N = 16   #Limiting for speed purposes: Should be left as above
    X = np.zeros((N,R,K))
    
    ## Test first what happens if no restrictions on condition number 
    A = np.random.rand(N,N,K)
    for i in range(K) :
        X[:,:,i] = np.dot(A[:,:,i], loadmat(sys.argv[i])['SM'][0:N,:])
    
    test4(X, A, save_name="zeta_randomA")
     
    ## Now test what happens if condition number is bounded by 3
    A = np.random.rand(N,N,K)
    for i in range(K) :
        A[:,:,i] = fix_cond(A[:,:,i], 3)
        X[:,:,i] = np.dot(A[:,:,i], loadmat(sys.argv[i])['SM'][0:N,:])
    
    test4(X, A, save_name="zeta_conditionedA")
