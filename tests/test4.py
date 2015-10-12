import numpy as np
import matplotlib.pyplot as plt

from joint_isi import joint_ISI
from IVA_L import iva_l


def test4(X, A, verbose=False) :
    '''
    Sees how well normal IVA would do with subjects
        added one by one to a single site
    '''
    N,R,K = X.shape
    values = []
    for k in range(1,K/2) :
        A_m = A[:,:,0:2*k]
        X_m = np.zeros((N,R,k))
        X_m = X[:,:,0:2*k]
        W,_,_ = iva_l(X_m, verbose=False)
        values.append( (2*k, joint_ISI(W,A_m)) )

    X_values   = [value_pair[0] for value_pair in values]
    ISI_values = [value_pair[1] for value_pair in values]

    fil = open("test4_isi_val.csv", "w")
    fil.write("ISI_value,Number_sites\n")
    for a in range( len(values) ) :
        fil.write("%f,%f\n"%(X_values[a], ISI_values[a]))
    fil.close()
    
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of Sites")
    plt.xlabel("Number of Sites")
    plt.ylabel("ISI Value")
    plt.savefig("test4_picture")
    plt.close("all")

