import numpy as np
from IVA_L import iva_l
from joint_isi import joint_ISI
from scipy.io import loadmat
import matplotlib.pyplot as plt


def test1_2 () :
    K=10
    variables = loadmat("SCV_IVA_case12_r002_fakeA_sqcond3_nik.mat")
    A = np.zeros((16,16,K))
    for k in range(K) :
        A[:,:,k] = variables['A'][0,k]
    variables = loadmat("SCV_IVA_case12_r002.mat")
    X = np.zeros((16,32968,K))
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], variables['Sgt'][0,k][:,:])
    fil = open("test_SCV_r002.csv", "w")
    X_values = []
    ISI_values = []
    
    for k in range(2, K) :
        W,_,_ = iva_l(X[:,:,0:k], verbose=True)
        isi = joint_ISI(W,A[:,:,0:k])
        print isi
        fil.write("%i,%f"%(k,isi))
        X_values.append(k)
        ISI_values.append(isi)
    
    fil.close()
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of total Subjects")
    plt.xlabel("Number of Subjects")
    plt.ylabel("ISI Value")
    plt.savefig("normal_IVA_r002.pdf")
    plt.close("all")
 

test1_2()   
