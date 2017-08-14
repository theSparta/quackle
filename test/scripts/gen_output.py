import argparse
import subprocess
import numpy as np
import shlex

# for reproducibility
RANDOM_SEED = 42
DIR_NAME="/home/rishabh/Quackle/quackle/test/"
np.random.seed(RANDOM_SEED)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outfile")
    parser.add_argument("-i", "--num_iterations", type=int, default=50)
    parser.add_argument("-n", "--total_iterations", type=int, default=1000, required=True)
    parser.add_argument("-p", "--player2", default="Speedy Player")
    parser.add_argument("-w", "--weights", nargs=2, required=True)
    parser.add_argument("-e", "--executable", default="./test")
    parser.add_argument("--verbose", action="store_true", help="print output to terminal")
    parser.add_argument("--parallel", action="store_true",help="whether to use gnu parallel or not")

    args = parser.parse_args()
    use_parallel = args.parallel
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
    fn_weights = args.weights
    player = '"' + args.player2 + '"'
    verbose = args.verbose

    if args.outfile is None:
        outfile = DIR_NAME + "out/out_w_" + "_".join(fn_weights) + '_' + args.player2
    else:
        outfile = args.outfile

    command = "./test --mode=selfplay --lexicon=csw12 --computer2=" + \
            player +" --repetitions=" + str(num_iterations) + " --seed=" + \
            seed_arg + " --weights \"|" + "|".join(fn_weights) + '"'
    if use_parallel:
        command = "parallel '" + command  + "' ::: " +  " ".join(seeds)

    if verbose:
        print(command)
    args = shlex.split(command)
    p1 = subprocess.Popen(args, cwd=DIR_NAME,stdout=subprocess.PIPE)
    args = shlex.split("grep '^GAME'")
    p2 = subprocess.Popen(args, stdin=p1.stdout, stdout=subprocess.PIPE)
    args = shlex.split("awk '{print $7\",\"$NF}'")
    with open(outfile, 'w+') as f:
        p3 = subprocess.call(args, stdin=p2.stdout, stdout=f)
    exit_codes = [p.wait() for p in p1, p2]
