## Just general need modules
import numpy as np
import sys

def set_functions (X) :
    P = len(X)
    N,R = X[0].shape
    K = [Z.shape[2] for Z in X]
    Y = [Z.copy() for Z in X]
    
    W_master = [np.zeros((N,N,K[p])) for p in range(P)]    
    
    def cost_and_grad(W) :
        for p in range(P) :
            for k in range(K[p]) :
                W_master[p][:,:,:] = vec_to_mat(W[])
        for p in range(P) :
            for k in range(K[p]) :
                Y[p][:,:,k] = np.dot(W[p][:,:,k], X[p][:,:,k])


def diva_l (X, W_init, max_iter=1024, verbose=False) :
    '''
    IVA_L is the Independent Vector Analysis using multivariate Laplacian 
        distribution
    
    Inputs:
    -------
    X : List of 3-D arrays
        Has total list, length of which is total number of sites, each 
        element of which is a list of subjects within that site. Every 
        subject is a 2-D array, shape of which is (row, column).
    
        
    W_init : array, shape=(K,N,N), optional
        Initial guess for W? Defaults to np.random.rand(N,N,K), where N
        is number of rows of X, and K is number of rows deep X is 
        
        (ie X.shape = (N,T,K), and W.shape = (N,N,K))
    
    max_iter : Int, optional
        The maximum number of iterations the algorithm is allowed to run for.
        Defaults to 1024.
        
    verbose : Bool, optional
        Print iteration information to output? Defaults to False
    
    Outputs:
    --------
    W : list of arrays, shape = (N,N,K)
        The unmixing matrix W for each subject, ordered in order
        that subjects were recieved in
    
    '''
    

    P = len(X)
    N,R = X[0][0][:,:].shape
    
    if W_init == [] :
        W = [[np.random.rand(N, N) for x in range(len(X[p]))] for p in range(P)]
    else :
        W = W_init
    
    if (term_crit != 'ChangeInW') and (term_crit != 'ChangeInCost') :
        raise ValueError ('''term_crit has to be either 
                          'ChangeInW' or 'ChangeInCost' ''')
    
    ## Main Loop
            
    cost_and_grad = set_functions( X )
    
    W, _, d = fmin_l_bfgs_b(fun=cost_and_grad, x0=W_init)
    
    YtY = np.zeros(shape=(N,T,P))
            

    if iteration==max_iter :
        print ('''Algorithm may have not converged, reached max
               number of iterations ''')
    
    ## Finish display
    if verbose :
        print "Algorithim converged, end results are: "
        print " Step: %i \n W change: %f \n Cost %f \n\n" % (iteration, term_criterion, cost[iteration])
    
    return W, iteration, cost



class dIVA_L() :
    '''
    The class for dIVA_L, in a similar vein to the class for IVA_L
        or PCA or ICA
    '''
    
    def __init__(self, alpha0=0.1, term_threshold=1e-6, 
                 term_crit="ChangeInCost", max_iter=1024, W_init=[]) :
        self.alpha0         = alpha0
        self.term_threshold = term_threshold
        self.term_crit      = term_crit
        self.max_iter       = max_iter1024
        self.W_init         = W_init
        self.verbose        = verbose
    
    def fit(self, X) :
        '''
        Applies the fit to dIVA_L
        '''
        self.W, self.iteration, self.cost = diva_l(X)
    
    def transform(self, X) :    
        Y = [[np.zeros(shape=X.shape) for X in X[p]] for p in range(len(X))]
         
        for p in range(len(X)) :
            for k in range(len(X[p])) :
                Y[p][k][:,:] = np.dot(self.W[p][k][:,:], X[p][k][:,:])
         
        return Y

    
    def fit_transform(self, X) :
        
        self.fit(X)
        self.transform(X)

    
    
    
    