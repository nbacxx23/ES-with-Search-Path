#!/usr/bin/env python
"""Python script for the COCO experimentation module `cocoex`.

Usage from a system shell::

    python ES_with_Search_Path.py bbob

runs a full but short experiment on the bbob suite. The optimization
algorithm used is determined by the `SOLVER` attribute in this file.

    python ES_with_Search_Path.py bbob 20

Calling `ES_with_Search_Path` without parameters prints this
help and the available suite names.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
del absolute_import, division, print_function, unicode_literals
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import math
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass

def default_observers():
    """return a map from suite names to default observer names"""
    # this is a function only to make the doc available and
    # because @property doesn't work on module level
    return {'bbob':'bbob',
            'bbob-largescale':'bbob',  # todo: needs to be confirmed
            'bbob-constraint':'bbob',  # todo: needs to be confirmed
            'bbob-biobj': 'bbob-biobj'}

def print_flush(*args):
    """print without newline and flush"""
    print(*args, end="")
    sys.stdout.flush()


def ascetime(sec):
    """return elapsed time as str.

    Example: return `"0h33:21"` if `sec == 33*60 + 21`. 
    """
    h = sec / 60**2
    m = 60 * (h - h // 1)
    s = 60 * (m - m // 1)
    return "%dh%02d:%02d" % (h, m, s)


class ShortInfo(object):
    """print minimal info during benchmarking.

    After initialization, to be called right before the solver is called with
    the respective problem. Prints nothing if only the instance id changed.

    Example output:

        Jan20 18h27:56, d=2, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:56, d=3, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

        Jan20 18h27:57, d=5, running: f01f02f03f04f05f06f07f08f09f10f11f12f13f14f15f16f17f18f19f20f21f22f23f24f25f26f27f28f29f30f31f32f33f34f35f36f37f38f39f40f41f42f43f44f45f46f47f48f49f50f51f52f53f54f55 done

    """
    def __init__(self):
        self.f_current = None  # function id (not problem id)
        self.d_current = 0  # dimension
        self.t0_dimension = time.time()
        self.evals_dimension = 0
        self.evals_by_dimension = {}
        self.runs_function = 0
    def print(self, problem, end="", **kwargs):
        print(self(problem), end=end, **kwargs)
        sys.stdout.flush()
    def add_evals(self, evals, runs):
        self.evals_dimension += evals
        self.runs_function += runs
    def dimension_done(self):
        self.evals_by_dimension[self.d_current] = (time.time() - self.t0_dimension) / self.evals_dimension
        s = '\n    done in %.1e seconds/evaluation' % (self.evals_by_dimension[self.d_current])
        # print(self.evals_dimension)
        self.evals_dimension = 0
        self.t0_dimension = time.time()
        return s
    def function_done(self):
        s = "(%d)" % self.runs_function + (2 - int(np.log10(self.runs_function))) * ' '
        self.runs_function = 0
        return s
    def __call__(self, problem):
        """uses `problem.id` and `problem.dimension` to decide what to print.
        """
        f = "f" + problem.id.lower().split('_f')[1].split('_')[0]
        res = ""
        if self.f_current and f != self.f_current:
            res += self.function_done() + ' '
        if problem.dimension != self.d_current:
            res += '%s%s, d=%d, running: ' % (self.dimension_done() + "\n\n" if self.d_current else '',
                        ShortInfo.short_time_stap(), problem.dimension)
            self.d_current = problem.dimension
        if f != self.f_current:
            res += '%s' % f
            self.f_current = f
        # print_flush(res)
        return res
    def print_timings(self):
        print("  dimension seconds/evaluations")
        print("  -----------------------------")
        for dim in sorted(self.evals_by_dimension):
            print("    %3d      %.1e " %
                  (dim, self.evals_by_dimension[dim]))
        print("  -----------------------------")
    @staticmethod
    def short_time_stap():
        l = time.asctime().split()
        d = l[0]
        d = l[1] + l[2]
        h, m, s = l[3].split(':')
        return d + ' ' + h + 'h' + m + ':' + s

# ===============================================
# prepare (the most basic example solver)
# ===============================================
def random_search(fun, lbounds, ubounds, budget):
    """Efficient implementation of uniform random search between `lbounds` and `ubounds`."""
    lbounds, ubounds = np.array(lbounds), np.array(ubounds)
    dim, x_min, f_min = len(lbounds), (lbounds + ubounds) / 2, None
    max_chunk_size = 1 + 4e4 / dim
    while budget > 0:
        chunk = int(min([budget, max_chunk_size]))
        # about five times faster than "for k in range(budget):..."
        X = lbounds + (ubounds - lbounds) * np.random.rand(chunk, dim)
        F = [fun(x) for x in X]
        if fun.number_of_objectives == 1:
            index = np.argmin(F)
            if f_min is None or F[index] < f_min:
                x_min, f_min = X[index], F[index]
        budget -= chunk
    return x_min

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
    
    x_final = np.sum(x_k, axis=0)/lamda                     # the final solution vector after the optimization process

    h = 0                                                   # stop criterion--if happy h=1,otherwise h=0
    
    E_half_normal_dis = math.sqrt(2/math.pi)                # parameter used to update the step-size   
    
    E_muldim_normal = math.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    
    while h==0 and budget>0 :
        z = np.random.multivariate_normal(mean, I, lamda)

        x = np.array(np.ones((lamda,n)))*x_final
        
        x_k = x + z*sigma

        p,z_u_best = sel_u_best(u,x_k,z,fun)                                     # select the u best solution to keep the population size constant
        
        "update the search path"
        s_sig = (1-c_sig)*s_sig + math.sqrt(c_sig*(2-c_sig))*(math.sqrt(u)/u)*(z_u_best.sum(axis=0))
        "update the step size"
        sigma = sigma * np.exp((1/d_i)*((abs(s_sig)/ E_half_normal_dis)-1))*math.exp((c_sig/d)*((np.linalg.norm(s_sig)/E_muldim_normal)-1))
        
        x = np.sum(p,axis = 0)/u

        h = happy(x, x_final,fun)
        
        x_final = x

        budget = budget - 1
       #print x,fun(x_final),budget,h
    return x,fun(x_final),f_evaluation-budget,h
        
# ===============================================
# recombination and parent update  
# =============================================== 
def sel_u_best(u, x_k,z_k,fun):
    lamda = len(x_k)              #x_k has the same length of the offspring population size
    f_x = np.zeros(lamda)         #objective function
    u_best = [[] for i in range(u)]
    z_k_best = [[] for i in range(u)]
    for i in xrange(lamda):
        f_x[i] = fun(x_k[i])
    sorted_f_x = sorted(xrange(lamda), key=lambda k: f_x[k])  #rank the solution based on their objective function value
    #print u
    for i in xrange(u):
        u_best[i] = x_k[sorted_f_x[i]]
        z_k_best[i] = z_k[sorted_f_x[i]]
    u_best = np.array(u_best)
    z_k_best = np.array(z_k_best)
    return u_best,z_k_best

# ===============================================
# the stop criterion  
# ===============================================
def happy(x,x_final,fun):
    """
    Setting up a threshold which decides the stop criterion. 
    When the difference of the last two resulte solutions is not bigger than the threshold, we consider
    the resulte arrives the convergence(happy)
    """
    if abs(fun(x)-fun(x_final)) <= 0.000000001 :
        return 1
    else:
        return 0        

# ===============================================
# loops over a benchmark problem suite
# ===============================================
def batch_loop(solver, suite, observer, budget,
               max_runs, current_batch, number_of_batches):
    """loop over all problems in `suite` calling
    `coco_optimize(solver, problem, budget * problem.dimension, max_runs)`
    for each eligible problem.

    A problem is eligible if
    `problem_index + current_batch - 1` modulo `number_of_batches`
    equals to zero.
    """
    addressed_problems = []
    short_info = ShortInfo()
    for problem_index, problem in enumerate(suite):
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        observer.observe(problem)
        short_info.print(problem) if verbose else None
        runs = coco_optimize(solver, problem, budget * problem.dimension, max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations, runs)
        problem.free()
        addressed_problems += [problem.id]
    print(short_info.function_done() + short_info.dimension_done())
    short_info.print_timings()
    print("  %s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")
    if number_of_batches > 1:
        print("\n    MAKE SURE TO RUN ALL BATCHES", end="")
    return addressed_problems

#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, max_evals, max_runs=1e9):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until either the `max_evals` are exhausted or `max_run` solver calls
    have been made or the `solver` has not called `fun` even once
    in the last run.

    Return number of (almost) independent runs.
    """
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    if fun.evaluations:
        print('WARNING: %d evaluations were done before the first solver call' %
              fun.evaluations)

    for restarts in range(int(max_runs)):
        remaining_evals = max_evals - fun.evaluations
        x0 = center + (restarts > 0) * 0.8 * range_ * (
                np.random.rand(fun.dimension) - 0.5)
        fun(x0)  # can be incommented, if this is done by the solver

        if solver.__name__ in ("random_search", ):
            solver(fun, fun.lower_bounds, fun.upper_bounds,
                   remaining_evals)
        elif solver.__name__ == 'fmin' and solver.__globals__['__name__'] in ['cma', 'cma.evolution_strategy', 'cma.es']:
            if x0[0] == center[0]:
                sigma0 = 0.02
                restarts_ = 0
            else:
                x0 = "%f + %f * np.random.rand(%d)" % (
                        center[0], 0.8 * range_[0], fun.dimension)
                sigma0 = 0.2
                restarts_ = 6 * (observer_options.find('IPOP') >= 0)

            solver(fun, x0, sigma0 * range_[0], restarts=restarts_,
                   options=dict(scaling=range_/range_[0], maxfevals=remaining_evals,
                                termination_callback=lambda es: fun.final_target_hit,
                                verb_log=0, verb_disp=0, verbose=-9))
        elif solver.__name__ == 'fmin_slsqp':
            solver(fun, x0, iter=1 + remaining_evals / fun.dimension,
                   iprint=-1)
############################ ADD HERE ########################################
        # ### IMPLEMENT HERE THE CALL TO ANOTHER SOLVER/OPTIMIZER ###
        # elif True:
        #     CALL MY SOLVER, interfaces vary
##############################################################################
        elif solver.__name__ == 'ES_search_path':
            solver(fun,fun.lower_bounds,fun.upper_bounds,remaining_evals)

        else:
            raise ValueError("no entry for solver %s" % str(solver.__name__))

        if fun.evaluations >= max_evals or fun.final_target_hit:
            break
        # quit if fun.evaluations did not increase
        if fun.evaluations <= max_evals - remaining_evals:
            if max_evals - fun.evaluations > fun.dimension + 1:
                print("WARNING: %d evaluations remaining" %
                      remaining_evals)
            if fun.evaluations < max_evals - remaining_evals:
                raise RuntimeError("function evaluations decreased")
            break
    return restarts + 1

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
budget = 2  # maxfevals = budget x dimension ### INCREASE budget WHEN THE DATA CHAIN IS STABLE ###
max_runs = 1e9  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################
SOLVER = ES_search_path
#SOLVER = my_solver # fmin_slsqp # SOLVER = cma.fmin
suite_name = "bbob-biobj"
# suite_name = "bbob"
suite_instance = "year:2016"
suite_options = ""  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
observer_name = default_observers()[suite_name]
observer_options = (
    ' result_folder: %s_on_%s_budget%04dxD '
                 % (SOLVER.__name__, suite_name, budget) +
    ' algorithm_name: %s ' % SOLVER.__name__ +
    ' algorithm_info: "AN EVOLUTION STRATEGIES WITH SEARCH PATH" ') 
######################### END CHANGE HERE ####################################

# ===============================================
# run (main)
# ===============================================
def main(budget=budget,
         max_runs=max_runs,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    `batch_loop(SOLVER, suite, observer, budget,...`.
    """
    observer = Observer(observer_name, observer_options)
    suite = Suite(suite_name, suite_instance, suite_options)
    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.clock()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." % (time.asctime(), ascetime(time.clock() - t0)))

# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            print("Recognized suite names: " + str(cocoex.known_suite_names))
            exit(0)
    suite_name = sys.argv[1]
    observer_name = default_observers()[suite_name]
    if len(sys.argv) > 2:
        budget = float(sys.argv[2])
        if observer_options.find('budget') > 0:  # reflect budget in folder name
            idx = observer_options.find('budget')
            observer_options = observer_options[:idx+6] + \
                "%04d" % int(budget + 0.5) + observer_options[idx+10:]
    if len(sys.argv) > 3:
        current_batch = int(sys.argv[3])
    if len(sys.argv) > 4:
        number_of_batches = int(sys.argv[4])
    if len(sys.argv) > 5:
        messages = ['Argument "%s" disregarded (only 4 arguments are recognized).' % sys.argv[i]
            for i in range(5, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget, max_runs, current_batch, number_of_batches)
