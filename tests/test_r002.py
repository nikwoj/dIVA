import numpy as np
from IVA_L import IVA_L
from joint_isi import joint_ISI
from scipy.io import loadmat
import matplotlib.pyplot as plt


def test1_2 () :
    variables = loadmat("SCV_IVA_case12_r002.mat")
    S = variables['S']
    A = variables['A']
    N,R,K = S.shape
    X = np.zeros((N,R,K))
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], S[:,:,k])
    rm(S)
    fil = open("test_SCV_r002.csv", "w")
    
    for k in range(2, K) :
        W = IVA_L(X[:,:,0:k], verbose=True)
        fil.write("%i,%f"%(k,joint_ISI(W,A[:,:,0:k])))
        X_values.append(k)
        ISI_values.append(joint_ISI(W,A[:,:,k]))
    
    fil.close()
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of total Subjects")
    plt.xlabel("Number of Subjects")
    plt.ylabel("ISI Value")
    plt.savefig("Normal IVA r002")
    plt.close("all")
    
