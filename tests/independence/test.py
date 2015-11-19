import numpy as np
from numpy.random import rand
from measure_indep import measure_indep

if __name__=="__main__" :
    S = rand(5,100,6)
    measure_indep(S, "test", testing=True)
