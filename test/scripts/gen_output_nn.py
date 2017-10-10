import argparse
import subprocess
import numpy as np
import shlex
import sys
import os
import multiprocessing
import time

# for reproducibility
RANDOM_SEED = 42
DIR_NAME = os.path.dirname(os.getcwd())
np.random.seed(RANDOM_SEED)
num_proc = multiprocessing.cpu_count()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outfile")
    parser.add_argument("-i", "--num_iterations", type=int, default=50)
    parser.add_argument("-n", "--total_iterations", type=int, default=1000, required=True)
    parser.add_argument("-p", "--player2", default="Speedy Player")
    parser.add_argument("-a", "--player1", default = "Speedy Player")
    parser.add_argument("-e", "--executable", default="./test")
    parser.add_argument("-w", "--num_w", default=8)
    parser.add_argument("--verbose", action="store_true", help="print output to terminal")
    parser.add_argument("--parallel", action="store_true",help="whether to use gnu parallel or not")

    args = parser.parse_args()
    use_parallel = args.parallel
    num_w = args.num_w
    total_iterations = args.total_iterations
    if use_parallel:
        num_iterations = args.num_iterations
        assert(total_iterations % num_iterations == 0)
        num_proc = total_iterations//num_iterations
        seeds = [str(i) for i in np.random.randint(1000, size=num_proc)]
        seed_arg = "{1}"
    else:
        seed_arg = str(RANDOM_SEED)
        num_iterations = total_iterations
    exec_file = args.executable
    player1 = '"' + args.player1 + '"'
    player2 = '"' + args.player2 + '"'
    verbose = args.verbose
    num_position = 5 + len(args.player1.split())

    if args.outfile is None:
        outfile = os.path.join(DIR_NAME, "out/out_wnn_" + \
            args.player1 + '_' + args.player2 +  '_' + time.ctime() )
    else:
        outfile = args.outfile

    command = "./test --mode=selfplay --lexicon=csw12" + \
            " --computer={} --computer2={}".format(player1, player2) + \
            " --repetitions=" + str(num_iterations) + " --seed=" + \
            seed_arg + " -z {} ".format(num_w) 
    if use_parallel:
        command = "parallel '" + command  + "' ::: " +  " ".join(seeds)

    if verbose:
        print(command)
    args = shlex.split(command)
    p1 = subprocess.Popen(args, cwd=DIR_NAME, stdout=subprocess.PIPE)
    args = shlex.split("grep '^GAME'")
    p2 = subprocess.Popen(args, stdin=p1.stdout, stdout=subprocess.PIPE)
    args = shlex.split("awk '{print $#\",\"$NF}'")
    args[1] =  args[1].replace("#", str(num_position))
    with open(outfile, 'w+') as f:
        p3 = subprocess.call(args, stdin=p2.stdout, stdout=f)
    exit_codes = [p.wait() for p in p1, p2]
