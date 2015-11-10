import numpy as np
from IVA_L import iva_l
from joint_isi import joint_ISI
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys

def test1_2 (fil1, number) :
    K=10
    A = np.random.rand(16,16,K)
    variables = loadmat(fil1)
    X = np.zeros((16,32968,K))
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], variables['Sgt'][0,k][:,:])
    fil = open("test_SCV_r0%i.csv"%number, "w")
    fil.write("Iteration,ISI\n")
    X_values = []
    ISI_values = []
    
    for k in range(2, K) :
        W,_,_ = iva_l(X[:,:,0:k], verbose=True)
        isi = joint_ISI(W,A[:,:,0:k])
        print isi
        fil.write("%i,%f\n"%(k,isi))
        X_values.append(k)
        ISI_values.append(isi)
   
    fil.close()
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of total Subjects")
    plt.xlabel("Number of Subjects")
    plt.ylabel("ISI Value")
    plt.savefig("normal_IVA_r0%i.pdf"%number)
    plt.close("all")
 

if __name__ == "__main__" : 
    sys.argv.pop(0)
    l = len(sys.argv)
    for i in range(l) :
        test1_2(sys.argv[i], i)
