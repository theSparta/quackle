import sys
import cPickle as pickle
from os.path import dirname, abspath, join
import multiprocessing
from joblib import Parallel, delayed
from move import PyMove

def bad_move(move):
	return move[0][0] == '-'

def filtered_moves(movelist):
	moves = []
	for move in movelist:
		if not bad_move(move):
			moves.append(move)
	return movelist

def update(move_dict):
	for key, val in move_dict.items():
		updated_move = filtered_moves(val)
		if(len(updated_move) < 2):
			del move_dict[key]
		else:
			move_dict[key] = updated_move
	print(len(move_dict.keys()))

if __name__ == '__main__':

	PARENT_DIR= dirname(dirname(abspath(__file__)))
	DATA_FILE = join(PARENT_DIR, 'test/expert/Ninety_Second_Championship_Player_best_moves.p')
	NEW_DATA_FILE = join(PARENT_DIR, 'test/expert/Filtered_championship_moves.p')
	print("Loading moves file...")
	move_dict = pickle.load(open(DATA_FILE, 'r'))
	print("Updating moves file...")
	update(move_dict)
	print("Saving moves file in {}...".format(NEW_DATA_FILE))
	pickle.dump(move_dict, open(NEW_DATA_FILE, 'w'))