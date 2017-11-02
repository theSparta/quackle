import numpy as np
import multiprocessing as mp
from joblib import Parallel, delayed
import cPickle as pickle

N_CPU = mp.cpu_count()

def convert_board(board):
    N = board.shape[0]
    num_maps = 27
    feature_maps = np.empty((N, N, num_maps), dtype='bool')
    chars = [' ', "=", "'"]
    feature_maps[:,:,0] = (board == chars[0])
    for c in chars[1:]:
        feature_maps[:,:,0] |= (board == c)
    for i in range(1, num_maps):
        feature_maps[:,:,i] = (board == chr(ord('A') + i-1)) | \
            (board == chr(ord('a')+ i - 1))
    return feature_maps.astype('int8')


def convert_frac(board_args):
    keys = board_args.keys()
    mydict = {}
    for key in keys:
        init, boards = board_args[key]
        boards = [convert_board(board) for board in boards]
        mydict[key] = tuple((convert_board(init), boards))
    return mydict

def merge_dicts(dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def convert_boards(boards):
    keys = boards.keys()
    frac = (len(keys) // N_CPU) + 1
    with Parallel(n_jobs = N_CPU) as parallel:
        fracs = [i*frac for i in range(N_CPU)]
        boards_args = [{k: boards[k] for k in keys[fracs[i]:fracs[i]+frac]}
                       for i in range(N_CPU)]
        dicts = parallel(delayed(convert_frac)(boards_args[i]) for
                         i in range(N_CPU))
    return merge_dicts(dicts)

if __name__ == '__main__':
    boards_file = "boards_dict.p"
    boards = pickle.load(open(boards_file, "rb"))
    keys = boards.keys()
    frac = (len(keys) // N_CPU) + 1
    with Parallel(n_jobs = N_CPU) as parallel:
        fracs = [i*frac for i in range(N_CPU)]
        boards_args = [{k: boards[k] for k in keys[fracs[i]:fracs[i]+frac]}
                       for i in range(N_CPU)]
        dicts = parallel(delayed(convert_frac)(boards_args[i]) for
                         i in range(N_CPU))
    merge_dicts(dicts)
#   convert_boards(boards_map)
    print("Done")
