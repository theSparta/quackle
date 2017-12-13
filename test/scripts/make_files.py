import sys
import os
import glob
import cPickle as pickle

CURR_DIR = '/home/rishabh/quackle/test'
START_LINES = 3
EXTENSION = '.gcg'

def make_rack(line, turn):
    vals = line.split(' ')
    rack = vals[1]
    if not rack:
        return None, None
    else:
	move = vals[2:4]
    return "#rack{} {}\n".format(turn, rack), move


def create_file(file, dirname, counter, move_made):

    filename, extension = os.path.splitext(file)
    file_name = os.path.join(dirname, file)
    with open(file_name, 'r') as f:
        lines = f.readlines()

    num_files = len(lines) - 2 * START_LINES
    filenames = ['{}#{}#{}{}'.format(filename, i, counter, extension) for
                 i in range(num_files)]
    file_ptrs = [open(os.path.join(DIRNAME, file) , 'w') for file in filenames]
    # print(filenames)

    for line in lines[:START_LINES]:
        for f in file_ptrs:
            f.write(line)

    lines = lines[START_LINES:]

    for i, line in enumerate(lines):
        if i == num_files:
            break
        rack, move = make_rack(line, 2 -i%2)
        if rack is not None:
            file_ptrs[i].write(rack)
        move_made[filenames[i]] = move
        file_ptrs[i].close()
        for f in file_ptrs[i+1:]:
            f.write(line)

def create_files():
    move_made = {}
    dir_regex = os.path.join(CURR_DIR, "games_Five_Minute_Championship_Player*")
    counter = 0
    for dirname in glob.glob(dir_regex):
        counter += 1
        for file in os.listdir(dirname):
            if file.endswith(EXTENSION):
                create_file(file, dirname, counter, move_made)
                print("%s done" % file)
    return move_made


if __name__ == '__main__':

    DIRNAME = os.path.join(CURR_DIR, 'gcg')
    if not os.path.exists(DIRNAME):
        os.makedirs(DIRNAME)
    move_made = create_files()
    pickle.dump(move_made, open('moves_games.pickle', 'w'))

# create_files(file)
