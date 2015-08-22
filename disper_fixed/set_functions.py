import numpy as np

from numpy.linalg import pinv, det
from vec_mat import vec_to_mat, mat_to_vec

def set_functions_opt(list_of_sites, verbose, list_of_W) :
    '''
    Used during the optimization process to update the dispersion matrix
    '''
    
    p = 0
    list_of_disper
    for site in list_of_sites :
        site._initialize(verbose, list_of_W[p]) 
    
    