import sys
import cPickle as pickle
from os.path import dirname, abspath, join
import multiprocessing
from joblib import Parallel, delayed
from move import PyMove


PARENT_DIR= dirname(dirname(abspath(__file__)))
DATA_FILE = join(PARENT_DIR, 'test/expert/Filtered_championship_moves.p')
GAMES_DIR = join(PARENT_DIR, 'test/gcg')
CPUs = multiprocessing.cpu_count()


print("Loading moves file...")
moves = pickle.load(open(DATA_FILE, 'rb'))
files = sorted(moves.keys())
print("{} game files to be read".format(len(files)))

PyMove.init()

def get_features(file):
    pymove = PyMove()
    pymove.setGame(join(GAMES_DIR, file))
    features = []
    # init_board = pymove.board()
    # print(init_board)
    # boards = []
    for move in moves[file]:
        feature = pymove.getFeatures(move[0], move[1])
        # board = pymove.boardAfterMoveMade()
        # boards.append(board)
        features.append(feature)
    print("{} done!".format(file))
    return (file, features)

print("Getting features...")
with Parallel(n_jobs = CPUs) as parallel:
    moves_tuples = parallel(delayed(get_features)(file) for file in files)
    moves_dict = dict(moves_tuples)

print("Saving features in moves_dict.p...")
pickle.dump(moves_dict, open('moves_dict.p', 'wb'))
