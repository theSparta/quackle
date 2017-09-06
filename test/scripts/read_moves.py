import os
import argparse
import subprocess
import numpy as np
import shlex
import sys
import cPickle as pickle
import multiprocessing

# for reproducibility
RANDOM_SEED = 42
DIR_NAME="/home/ubuntu/quackle/test"
np.random.seed(RANDOM_SEED)

def parsemove(line):
    return (line.split())[:2]

def merge_dicts(dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def getMoves(command):
    cmd_args = shlex.split(command)
    p1 = subprocess.Popen(cmd_args, cwd=DIR_NAME,stdout=subprocess.PIPE, stderr=DEV_NULL)
    moves = {}
    counter = -1

    for line in p1.stdout:
	line = line.strip()
        startChar = line[0]
        if startChar == '#':
            counter = -1
        elif startChar == '$':
            counter = 0
            key = line.split(" ")[-1][skip_len:]
            moves[key] = []

        if counter == 0 or counter == 1 :
            counter += 1
        elif counter == 2:
            moves[key].append(parsemove(line))

    return moves

if __name__ == '__main__':

    DEV_NULL = open('/dev/null', 'w')
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outfile", default="Speedy_Moves.pickle")
    parser.add_argument("-p", "--player", default="Speedy Player")
    parser.add_argument("-e", "--executable", default="./test")
    parser.add_argument("-d", "--directory", default = "gcg")
    parser.add_argument("-w", "--weights", nargs=2, required=True)
    parser.add_argument("--verbose", action="store_true", help="print output to terminal")
    parser.add_argument("--parallel", action="store_true",help="whether to use gnu parallel or not")
    parser.add_argument("--save", action="store_true", help="to save the moves to outfile")
    args = parser.parse_args()
    use_parallel = args.parallel
    num_parallel = 16
    if use_parallel:
        seeds = [str(i) for i in np.random.randint(1000, size=num_parallel)]
        seed_arg = "{1}"
    else:
        seed_arg = str(RANDOM_SEED)
    exec_file = args.executable
    player = '"' + args.player + '"'
    verbose = args.verbose

    if args.outfile is None:
        outfile = DIR_NAME + "moves"
    else:
        outfile = args.outfile

    command = "./test --mode=positions --lexicon=csw12 --computer=" + \
            player  + " --weights \"|" + \
            "|".join(args.weights) + '" '

    dirname = os.path.join(DIR_NAME, args.directory)
    skip_len = len(args.directory) + 1
    files =  sorted(os.listdir(dirname))
    strs = [ "--position {}".format(os.path.join(args.directory, file))
	       for file in files]

    if use_parallel:
        pool = multiprocessing.Pool(num_parallel)
        commands = [command + " --seed=42 " +  " ".join(strlist)
                for strlist in chunks(strs, (len(strs)//num_parallel) + 1)]
        dicts = pool.map(getMoves,  commands)
        moves = merge_dicts(dicts)
    else:
        command = command + " ".join(strs) 
        moves = getMoves(command)

    if verbose:
        print(command)
    	for key in sorted(moves.keys()):
            print("key {}".format(key))
    	    for k in moves[key][:2]:
    	    	print(k)
    if args.save:
    	outfile = os.path.join(DIR_NAME, args.outfile)
    	print("Saving to file %s" %(outfile,))
    	pickle.dump(moves, open(outfile, 'wb'))
    DEV_NULL.close()
