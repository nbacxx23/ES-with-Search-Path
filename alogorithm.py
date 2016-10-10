# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:39:04 2016

The (u/u,lamda) - Evolution strategy with Search Path

@author: Yuxiang Wang/Xiaoxiao CHEN

"""

import math
import numpy as np
import sys
# ===============================================
# the main structure of the algorithm  
# ===============================================
def ES_search_path(fun, lbounds, ubounds, budget):
    "ES with cumulative path length control"
    lbounds,ubounds = np.array(lbounds),np.array(ubounds)   #define the search range
    n = len(lbounds)                                        #the dimension of the solution
    lamda = 20                                              #initialize lambda - offspring population size
    u = int(lamda/4)                                        #initialize parental population size
    "parameters used to update the  step-size and search path"
    c_sig = math.sqrt(u/(u+n))
    d = 1 + math.sqrt(u/n)
    d_i = 3*n
    
    f_evaluation = budget
    
    "initialize all the variables"
    s_sig = np.zeros(n)                                     # initialize search path - carries info about the interrelation between single steps
    mean = np.zeros(n) 
    I = np.eye(n)
    sigma = np.ones(n)*0.1                                  # initialize the step-size
    x_k = np.random.uniform(lbounds, ubounds,(lamda,n))     # initialize the solution(x-vector)
    
    x_final =np.sum(x_k, axis=0)/u                          # the final solution vector after the optimization process
    h = 0                                                   # stop criterion--if happy h=1,otherwise h=0
    E_half_normal_dis = math.sqrt(2/math.pi)                # parameter used to update the step-size   
    E_muldim_normal = math.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    
    while h==0 and budget>0 :
        z = np.random.multivariate_normal(mean, I, (lamda, n))

        for i in xrange(lamda):
            x_k[i] = x_final + z[i]*sigma
        p = np.concatenate((x_k, sel_u_best(u,x_k)), axis = 0)   # add the selected parents to the original population to complete recombination and parent update
        p = sel_u_best(u, p)                                     # select the u best solution to keep the population size constant
        
        "update the search path"
        s_sig = (1-c_sig)*s_sig + math.sqrt(c_sig*(2-c_sig))*(math.sqrt(u)/u)*(z.sum(axis=0))
        "update the step size"
        sigma = sigma * np.exp((1/d_i)*((abs(s_sig)/ E_half_normal_dis)-1))*math.exp((c_sig/d)*((np.linalg.norm(s_sig)/E_muldim_normal)-1))
        
        x = np.sum(p,axis = 0)/u
        h = happy(x, x_final)
        x_final = x
        budget = budget - 1
        print x,fun(x_final),budget,h
    return x,fun(x_final),f_evaluation-budget,h
        
# ===============================================
# recombination and parent update  
# =============================================== 
def sel_u_best(u, x_k):
    lamda = len(x_k)              #x_k has the same length of the offspring population size
    f_x = np.zeros(lamda)         #objective function
    u_best = [[] for i in range(u)]
    for i in xrange(lamda):
        f_x[i] = fun(x_k[i])
    sorted_f_x = sorted(xrange(lamda), key=lambda k: f_x[k])  #rank the solution based on their objective function value
    for i in xrange(u):
        u_best[i] = x_k[sorted_f_x[i]]
    u_best = np.array(u_best)
    return u_best

# ===============================================
# the stop criterion  
# ===============================================
def happy(x,x_final):
    """
    Setting up a threshold which decides the stop criterion. 
    When the difference of the last two resulte solutions is not bigger than the threshold, we consider
    the resulte arrives the convergence(happy)
    """
    if abs(fun(x)-fun(x_final)) <= 0.0000001 :
        return 1
    else:
        return 0        
        
# ===============================================
# the objective function(offered by default in coco)
# ===============================================
def fun(x):
    return x*x*5+1
    
# ===============================================
# test function  
# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) == 2:
        budget = float(sys.argv[1])
    ES_search_path(fun, [-1], [5], budget)