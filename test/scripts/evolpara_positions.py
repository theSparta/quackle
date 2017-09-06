from __future__ import print_function
import numpy as np
from joblib import Parallel, delayed
import sys
import cPickle as pickle
import shlex
import os
import subprocess
from cma import CMAEvolutionStrategy
from cma.utilities.utils import pprint

DIR_NAME = "/home/ubuntu/quackle/test"
#MULTIPLIER = 1e5
#GAMESPERAGENT = 50000

def fitness_val(filename, ideal_moves):
    moves = pickle.load(open(filename, 'rb'))
    fitness = 0.0
    for key in moves:
        try:
            ideal_move = ideal_moves[key]
        except KeyError:
            continue
        for i, move in enumerate(moves[key]):
            if (move == ideal_move):
                fitness -= 1./(i+1)
                break
    return fitness

def fitnessfunc(weights, counter, ideal_moves):
    ws = map(str, weights)
    out_name = os.path.join(DIR_NAME, 'moves/out_w_' + "_".join(ws))
    cmd = "python read_moves.py --parallel --save -o {} --weights {}".format(out_name, " ".join(ws))
    p = subprocess.call(shlex.split(cmd), cwd = os.path.join(DIR_NAME, 'scripts'))
    fitness = fitness_val(out_name, ideal_moves)
    print("Generation " + str(itercounter) + "_" + str(counter) + " weights = " + np.array_str(weights) + " Fitness = "+str(fitness))
    sys.stdout.flush()
    return fitness

if __name__=="__main__":

    moves_file = os.path.join(DIR_NAME, 'scripts/moves_games.pickle')
    ideal_moves = pickle.load(open(moves_file, 'rb'))
    #PARAMS
    popsize = 30
    maxiter = 4 #6
    itercounter = 0 # Number for new gen start

    initweights = [ 1. , 0.95]
    es = CMAEvolutionStrategy(initweights, 0.1, {'popsize':popsize, 
	'maxiter':maxiter, 'bounds': [0, 1], 'fixed_variables':{0:1.0}})
    #ALTERNATIVELY, load
    #es = pickle.load(open('saved-cma-object_' + str(itercounter) + '.pkl', 'rb'))

    with Parallel (n_jobs=4) as parallel:
        while not es.stop():
            print("Generation " + str(itercounter+1))
            sys.stdout.flush()
            solutions = es.ask()
            fitnessvector = [fitnessfunc(weights, counter, ideal_moves)
                    for (weights, counter) in
                    zip (solutions, xrange(1, popsize+1)) ]
            # print(fitnessvector)
            es.tell(solutions, fitnessvector)
            es.disp()  # uses option verb_disp with default 100
            sys.stdout.flush()

            print("Hello SUCKER!")
            es.result_pretty()
            sys.stdout.flush()

            # esfile =  DIR_NAME + 'scripts/cma_out/saved-fitness-cma-object_' + str(itercounter+1) + '.pkl'
            # pickle.dump(es, open(esfile, 'wb'))
            # print('Saved weights at end of generation ' + str(itercounter) + " in file " + esfile)

            itercounter += 1

        pprint(es.best.__dict__)
        sys.stdout.flush()

        #es.logger.add()  # write data to disc to be plotted
        #es.disp()

