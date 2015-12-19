import numpy as np
from IVA_L import iva_l
from joint_isi import joint_ISI
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sys

def test1_2 (fil1, fil2, number) :
    K=10
    variables = loadmat(fil1)
    A = np.zeros((16,16,K))
    for k in range(K) :
        A[:,:,k] = variables['A'][0,k]
    
    variables = loadmat(fil2)
    X = np.zeros((16,32968,K))
    for k in range(K) :
        X[:,:,k] = np.dot(A[:,:,k], variables['Sgt'][0,k][:,:])
    
    fil = open("test_SCV_r0%i_FA.csv"%number, "w")
    fil.write("Iteration,ISI\n")
    fil2 = open("temp_ISI.txt", "w")
    X_values = []
    ISI_values = []
    
    for k in range(2, K) :
        W,_,_ = iva_l(X[:,:,0:k], verbose=True)
        
        isi = joint_ISI(W,A[:,:,0:k])
        print isi
        
        fil.write("%i,%f\n"%(k,isi))
        fil2.write("%i"%isi)
        
        X_values.append(k)
        ISI_values.append(isi)
   
    fil.close()
    fil2.close()
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of total Subjects")
    plt.xlabel("Number of Subjects")
    plt.ylabel("ISI Value")
    plt.savefig("normal_IVA_r0%i.pdf"%number)
    plt.close("all")
 

if __name__ == "__main__" : 
    sys.argv.pop(0)
    l = len(sys.argv)
    for i in range(0, l, 2) :
        test1_2(sys.argv[i], sys.argv[i+1], i/2.0)
