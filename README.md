# IVA
For implementation of normal and distributed IVA

We will now explain each of the folders:

algorithm : 
    Contains both distributed and normal IVA algorithm

tests : 
    Contains files used to test degree to which the algorithm works. 
    Currently four such files :
        test1 : Fix number of subjects per site, add more sites
        test2 : Fix number of sites, add more subjects per site
        test3 : Fix total number of subjects, increase number of sites, decrease number of sites
                Currently non-operational
        test4 : 1 site, increase number of subjects per site, run normal IVA algorithm
    
    tests also contains initial results, located in folders inside tests, which have the results
        when applied to different data sets.

vec_mat :
    Contains files used to transform matrices into vectors and vice-versa, since scipy's optimize
        function is incompatible with optimizing with respect to matrices

joint_isi :
    Contains files used to score final output of algorithm.

variables : 
    Contains basic multivariate laplacian random variables
        
