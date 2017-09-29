from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import sys
import pickle
import shlex
import subprocess
import cma
import os

from cma import CMAEvolutionStrategy
from cma.utilities.utils import pprint


from multiprocessing import cpu_count

DIR_NAME = os.path.dirname(os.getcwd()) + "/"
MULTIPLIER = 1e5
AGENTS = cpu_count()
GAMESPERAGENT = 50000//AGENTS

def fitness_val(filename):
    scores = np.genfromtxt(filename, delimiter=',')
    wins = np.sum(np.argmax(scores, axis=1))
    total = scores.shape[0]
    totalScores = np.sum(scores, axis = 0)
    print("wins %d/%d" % (wins, total))
    # return -wins * MULTIPLIER +
    return -MULTIPLIER * wins + totalScores[0] - totalScores[1]

def fitnessfunc(weights, counter):
    ws = map(str, weights)
    out_name = DIR_NAME + 'out/out_w_' + "_".join(ws) + '_Speedy Player_Speedy Player' 
    cmd = "python gen_output.py --parallel -n {} -i {} --weights {}".format(
            str(GAMESPERAGENT * AGENTS), str(GAMESPERAGENT), " ".join(ws))
    p = subprocess.call(shlex.split(cmd), cwd=DIR_NAME + 'scripts')
    fitness = fitness_val(out_name)
    print("Generation " + str(itercounter) + "_" + str(counter) + " weights = " + np.array_str(weights) + " Fitness = "+str(fitness))
    return fitness

if __name__=="__main__":

    #PARAMS
    popsize = 25
    maxiter = 4
    itercounter = 0 # Number for new gen start

    initweights = [1.0, 1.0, 0.01 , 0 , 0 , 0 , 0 , 0 ]
    # Feb 24: this achieves 45 percent winrate
    es = CMAEvolutionStrategy(initweights, 0.1, {'popsize':popsize, 'maxiter':maxiter,
    'fixed_variables':{0:1.0}}) # 'bounds': [0, 1]
    #ALTERNATIVELY, load
    #es = pickle.load(open('saved-cma-object_' + str(itercounter) + '.pkl', 'rb'))

    #with Parallel (n_jobs=4) as parallel:
    logger = cma.CMADataLogger().register(es)
    while not es.stop():
        print("Generation " + str(itercounter+1))
        solutions = es.ask()
        fitnessvector =  [fitnessfunc(weights, counter) #parallel( delayed (
                for (weights, counter) in
                zip (solutions, xrange(1, popsize+1)) ]
        # print(fitnessvector)
        es.tell(solutions, fitnessvector)
        es.disp()  # uses option verb_disp with default 100

        print("Hello SUCKER!")
        es.result_pretty()

        # esfile =  DIR_NAME + 'scripts/cma_out/saved-fitness-cma-object_' + str(itercounter+1) + '.pkl'
        # pickle.dump(es, open(esfile, 'wb'))
        # print('Saved weights at end of generation ' + str(itercounter) + " in file " + esfile)

        itercounter += 1
        logger.add()

    pprint(es.best.__dict__)

    logger.plot()
    plt.savefig('cma_results_games')
    logger.figclose()

    #es.logger.add()  # write data to disc to be plotted
    #es.disp()

