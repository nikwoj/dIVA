import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

def measure_indep (S, save_name, testing=False) :
    '''
    Test the cross correlation and independence between various 
        components of the data set
    
    Inputs:
    =======
    S : Entire data set. Should be in Component, Samples, Subjects
        shape
    
    save_name : Desired name to save plot to. Should NOT end in .png
    
    Outputs:
    ========
    Nothing. Side effect is that a file with save_name.png will appear
        in working directory
    '''
    N,R,K = S.shape
    SS = np.zeros((N*K,R))
    for n in range(N) :
        for k in range(K) :
            SS[n*K + k, :] = S[n,:,k]
    SS = np.dot(SS, SS.T)
    
    if testing :
        SS_svd = svd(SS, compute_uv=False)
        print SS.max, SS_svd.max, SS_svd.mean
        
    heatmap = plt.pcolor(SS, cmap=plt.cm.gray)
    
    cbar = plt.colorbar(heatmap)
    plt.savefig(save_name + ".png")
    plt.close("all")
