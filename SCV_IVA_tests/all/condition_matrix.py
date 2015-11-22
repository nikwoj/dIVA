from numpy import dot
from numpy.linalg import svd
import sys

def fix_cond(A, max_condition=3.0) :
    '''
    Takes in square matrix and scales the singular
       values such that the maximum ratio between the 
       highest and lowest singular values is max_condition
    '''
    U, S, V = svd(A, compute_uv=True)
    # SVD orders singular values in order of highest to lowest
    alpha = (S[0] - max_condition * S[-1]) / (max_condition - 1 + sys.float_info.epsilon)
    S = S + alpha
    return dot(dot(U, S), V)
