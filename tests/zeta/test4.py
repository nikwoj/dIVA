import numpy as np
import matplotlib.pyplot as plt

from joint_isi import joint_ISI
from IVA_L import iva_l


def test4(X, A, save_name, step_size=2, verbose=True) :
    '''
    Sees how well normal IVA would do with subjects
        added step by step to a single site
    '''
    _,_,K = X.shape
    values = []
    fil = open(save_name + ".csv", "w")
    fil.write("ISI,subjects\n")
    
    X_values   = []
    ISI_values = []   
    for k in range(1, K / step_size) :
        tot_subjects = k * step_size
        W,_,_ = iva_l(X[:,:,0:tot_subjects], verbose=verbose)
        
        isi = joint_ISI(W,A[:,:,0:k*step_size])
        fil.write("%f,%i"%(isi, tot_subjects))
        
        X_values.append(tot_subjects)
        ISI_values.append(isi)
 
    fil.close()
    
    plt.plot(X_values, ISI_values)
    plt.title("ISI Value by Number of Sites")
    plt.xlabel("Number of Sites")
    plt.ylabel("ISI Value")
    plt.savefig(save_name + ".pdf")
    plt.close("all")

