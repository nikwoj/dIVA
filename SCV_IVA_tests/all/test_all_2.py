import numpy as np
from numpy.linalg import svd
from IVA_L import iva_l
from joint_isi import joint_ISI
from scipy.io import loadmat
from condition_matrix import fix_cond
import matplotlib.pyplot as plt
import sys

def test1_2 (fil1, A, number) :
    K=10
    variables = loadmat(fil1)
    X = np.zeros((16,32968,K))
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], variables['Sgt'][0,k][:,:])

    fil = open("test_SCV_r0%i.csv"%number, "w")
    for k in range(K) :
        A_SVD = svd(A[:,:,k], compute_uv=False)
        fil.write("Subject1: %i\n"%(A_SVD[0]/(sys.float_info.epsilon + A_SVD[-1])))  #Records the condition number of every subjects mixing matrix
    fil.write("Iteration,ISI\n")
    fil.close()
    
    X_values = []
    ISI_values = []
    
    for k in range(2, K) :
        W,_,_ = iva_l(X[:,:,0:k], verbose=True)
        isi = joint_ISI(W,A[:,:,0:k])
        
        fil = open("test_SCV_r0%i"%number, "a")
        fil.write("%i,%f\n"%(k,isi))
        fil.close()
        
        X_values.append(k)
        ISI_values.append(isi)
    
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of total Subjects")
    plt.xlabel("Number of Subjects")
    plt.ylabel("ISI Value")
    plt.savefig("normal_IVA_r0%i.pdf"%number)
    plt.close("all")


if __name__ == "__main__" : 
    sys.argv.pop(0)
    l = len(sys.argv)
    K = 10
    N,_ = loadmat(sys.argv[0])['Sgt'][0,0].shape #Need to know how big to make the mixing matrix
    A = np.random.rand(N,N,K)
    for i in range(l) :
        test1_2(sys.argv[i], A, i)
    
    for k in range(K) :
        A[:,:,k] = fix_cond(A[:,:,k])
    for i in range(l) :
        test1_2(sys.argv[i], A, i)
