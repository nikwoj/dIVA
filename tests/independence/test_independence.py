import numpy as np

import matplotlib.pyplot as plt


def test_indep (S, save_name) :
    '''
    Takes in data and returns graph displaying covariance
        based on components.
    
    Inputs
    ------
      S = Source signals
      
      save_name = Name to save plot to. Will be saved in pdf form,
         though save_name should not end in .pdf
    '''
    N,R,K = S.shape
    S2 = np.zeros((N*K,R)) # Make new array to load all components and subjects
    for n in range(N) :
        for nn in range(n+1) : 
            Cov = np.dot(S[n,:,:].T, S[nn,:,:])
            fig, ax = plt.subplots()
            heatmap = ax.pcolor(Cov, cmap=plt.cm.gray)
            
            # Put major ticks at middle of each cell
            ax.set_xticks(np.arange(Cov.shape[0]), minor=False)
            ax.set_yticks(np.arange(Cov.shape[1]), minor=False)
            
            # Want more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()
            #ax.title(str(n) + "-" + str(nn) + "Cross correlation")
            
            cbar = plt.colorbar(heatmap)
            plt.savefig(save_name + '%i-%i.png'%(n,nn))
    
