
import sys
import cPickle as pickle
from os.path import dirname, abspath, join
import multiprocessing
from joblib import Parallel, delayed
from move import PyMove
import numpy as np

PARENT_DIR= dirname(dirname(abspath(__file__)))
DATA_FILE = join(PARENT_DIR, 'test/expert/Filtered_championship_moves.p')
GAMES_DIR = join(PARENT_DIR, 'test/gcg')
CPUs = multiprocessing.cpu_count()

def process_board(boardStr):
    rows = boardStr.splitlines()[2:-1]
    rows = [row.split('|')[1] for row in rows]
    rows = [list(row[::2]) for row in rows]
    board = np.array(rows)
#    assert(board.shape == (15,15))
    return board

def get_features(file):
    pymove = PyMove()
    pymove.setGame(join(GAMES_DIR, file))
    features = []
    init_board = process_board(pymove.board())
    boards = []
    for move in moves[file]:
        feature = pymove.getFeatures(move[0], move[1])
        board = pymove.boardAfterMoveMade()
        boards.append(process_board(board))
        features.append(feature)
    print("{} done!".format(file))
    #print(process_board(boards[0]))
    return ((file, features), (file, tuple((init_board, boards))))


if __name__ == '__main__':

    print("Loading moves file...")
    moves = pickle.load(open(DATA_FILE, 'rb'))
    files = sorted(moves.keys())
    print("{} game files to be read".format(len(files)))
    PyMove.init()
    print("Getting features...")
    with Parallel(n_jobs = CPUs) as parallel:
        moves_tuples = parallel(delayed(get_features)(file) for file in files)
        moves_dict = dict((i[0] for i in moves_tuples))
        boards_dict = dict((i[1] for i in moves_tuples))

    moves_data_file = "moves_dict.p"
    boards_data_file = "boards_dict.p"
    print("Saving features, boards in {}, {}...".
        format(moves_data_file, boards_data_file))
    pickle.dump(moves_dict, open(moves_data_file, 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)
    pickle.dump(boards_dict, open(boards_data_file, 'wb'),
                protocol = pickle.HIGHEST_PROTOCOL)
    print("Done!")
