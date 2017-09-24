from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import sys
import pickle
import shlex
import subprocess
import os

from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}


DIR_NAME = os.path.dirname(os.getcwd()) + "/"
MULTIPLIER = 1e5
AGENTS = cpu_count()
GAMESPERAGENT = 500//AGENTS

def fitness_val(filename):
    scores = np.genfromtxt(filename, delimiter=',')
    wins = np.sum(np.argmax(scores, axis=1))
    total = scores.shape[0]
    totalScores = np.sum(scores, axis = 0)
    print("wins %d/%d" % (wins, total))
    # return -wins * MULTIPLIER +
    return MULTIPLIER * wins - totalScores[0] + totalScores[1]

def fitnessfunc(weights):
    ws = map(str, weights)
    out_name = DIR_NAME + 'out/out_w_' + "_".join(ws) + '_Speedy Player_Speedy Player'
    cmd = "python gen_output.py --parallel -n {} -i {} --weights {}".format(
            str(GAMESPERAGENT * AGENTS), str(GAMESPERAGENT), " ".join(ws))
    p = subprocess.call(shlex.split(cmd), cwd=DIR_NAME + 'scripts')
    fitness = fitness_val(out_name)
    return fitness

def target(w1, w2, w3, w4, w5, w6, w7, w8):
    return fitnessfunc([w1, w2, w3, w4, w5, w6, w7, w8])

if __name__ == '__main__':

    bo = BayesianOptimization(target,
                              {'w1': (1, 1), 'w2': (0.9, 1.1), 'w3':(-0.1, 0.1), 'w4':(-1, 1),
                              'w5': (-1, 1),'w6': (-1, 1), 'w7': (-1, 1), 'w8':(-1, 1)})

    # One of the things we can do with this object is pass points
    # which we want the algorithm to probe. A dictionary with the
    # parameters names and a list of values to include in the search
    # must be given.
    bo.explore({'w1': [1, 1, 1], 'w2': [1, 1, 1], 'w3': [0, 0.01, 0.02], 'w4':[0, 0, 0], 'w5':[0, 0, 0], 'w6':[0, 0, 0], 'w7':[0, 0, 0], 'w8':[0, 0, 0] })

    # Additionally, if we have any prior knowledge of the behaviour of
    # the target function (even if not totally accurate) we can also
    # tell that to the optimizer.
    # Here we pass a dictionary with 'target' and parameter names as keys and a
    # list of corresponding values
    bo.initialize(
        {
            'target' : [21800000, 21500000, 20800000],
            #'target': [2183937449.0, 2156350512.0, 2086549502.0],
            'w1': [1, 1, 1],
            'w2': [1, 1, 1],
            'w3': [0, 0.01, 0.02],
            'w4':[0, 0, 0],
            'w5':[0, 0, 0],
            'w6':[0, 0, 0],
            'w7':[0, 0, 0],
            'w8':[0, 0, 0]
        }
    )

    # Once we are satisfied with the initialization conditions
    # we let the algorithm do its magic by calling the maximize()
    # method.
    bo.maximize(init_points=5, n_iter=100, kappa=4, acq="ucb", **gp_params)

    # The output values can be accessed with self.res
    # print(bo.res['max'])

    # If we are not satisfied with the current results we can pickup from
    # where we left, maybe pass some more exploration points to the algorithm
    # # change any parameters we may choose, and the let it run again.
    # bo.explore({'x': [0.6], 'y': [-0.23]})

    # # Making changes to the gaussian process can impact the algorithm
    # # dramatically.
    # gp_params = {'kernel': None,
    #              'alpha': 1e-5}

    # # Run it again with different acquisition function
    # bo.maximize(n_iter=5, acq='ei', **gp_params)

    # # Finally, we take a look at the final results.
    print(bo.res['max'])
    print(bo.res['all'])

