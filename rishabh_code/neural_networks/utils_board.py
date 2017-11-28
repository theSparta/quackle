import numpy as np
from collections import Counter
import itertools
import cPickle as pickle
from convert_boards import convert_board

np.random.seed(42)

def shuffle(X, y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    X = X[randomize]
    y = y[randomize]
    return X, y

class DataGeneratorTest:
    """docstring for DataGenerator"""
    def __init__(self, boards, scores, num_train_frac = 0.98, batch_sz=512, load_from_file = False, 
                 save_to_file = False, file = "experience_buffer_boards.p"):
       
        assert(not(load_from_file and save_to_file))
        assert(sorted(boards.keys()) == sorted(scores.keys()))
       
        if load_from_file:
            npzfiles = np.load(file)
            X, y = npzfiles['X'], npzfiles['y']
            print("Data loaded from {}".format(file))
        else:
            X, y = [], []
            for key in boards:
                s, new_states = boards[key]
                assert(len(scores[key]) == len(scores[key]))
                X.extend([np.concatenate((s, s1), axis = -1) for s1 in new_states])
                y.extend(scores[key])
            X, y = shuffle(np.array(X), np.array(y))
            
            print(X.shape, y.shape)
            if save_to_file:
                np.savez(file, X=X, y=y)   # x,y,z equal sized 1D arrays
                print("Experience buffer saved to {}".format(file))
        
        num_train = int(num_train_frac * len(y))
        self.X_train, self.y_train = X[:num_train], y[:num_train]
        self.X_val, self.y_val = X[num_train:], y[num_train:]
        
        self.samples_per_train = num_train
        self.samples_per_val = len(X) - num_train
        print("Train: {} Val: {}".format(self.samples_per_train, self.samples_per_val))
        self.batch_sz = batch_sz
    
    def next_train(self):
        self.cur_train_index = 0
        while True:
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            X_batch = self.X_train[self.cur_train_index : self.cur_train_index + self.batch_sz]
            y_batch = self.y_train[self.cur_train_index : self.cur_train_index + self.batch_sz]
            self.cur_train_index += self.batch_sz
            yield (X_batch, y_batch)
    
    def next_val(self):
        self.cur_val_index = 0
        while True:
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            X_batch = self.X_val[self.cur_val_index : self.cur_val_index + self.batch_sz]
            y_batch = self.y_val[self.cur_val_index : self.cur_val_index + self.batch_sz]
            self.cur_val_index += self.batch_sz
            yield (X_batch, y_batch)
     
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

class BoardDataGenerator:
    """docstring for DataGenerator"""
    def __init__(self, boards, score_leaves, num_train_frac = 0.98, batch_sz=512, load_from_file = False, 
                 save_to_file = False, maxlen = 1, file = "experience_buffer.p"):
       
        assert(not(load_from_file and save_to_file))
        self.boards = boards
        self.scores = score_leaves
        self.keys = sorted(boards.keys())
        if load_from_file:
            experience_buffer = pickle.load(open(file, "rb"))
            print("Experience buffer loaded from {}".format(file))
        else:
            experience_buffer = []
            for i, key in enumerate(self.keys):
                arr = range(len(boards[key][1]))
                combinations = itertools.combinations(arr, 2)
                for pair in combinations:
                    if (pair[0] < maxlen and pair[1] >= maxlen):
                        experience_buffer.append((i, pair[0], pair[1]))
            experience_buffer = np.array(experience_buffer)
            print("Experience buffer generated")
            np.random.shuffle(experience_buffer)
        if save_to_file:
            pickle.dump(experience_buffer, open(file,"wb"), pickle.HIGHEST_PROTOCOL)
            print("Experience buffer saved to {}".format(file))
        
        num_train = int(num_train_frac * len(experience_buffer))
        self.experience_buffer_train = experience_buffer[:num_train]
        self.experience_buffer_val = experience_buffer[num_train:]
        
        self.samples_per_train = num_train
        self.samples_per_val = len(experience_buffer) - num_train
        print("Train: {} Val: {}".format(self.samples_per_train, self.samples_per_val))
        
        self.cur_train_index = 0
        self.cur_val_index = 0
        self.batch_sz = batch_sz
    
    def gen_pairs(self, experience_tuple_list):
        keys = self.keys
        boards, scores = [[], []], [[], []]
        boards[0] = ((self.boards[keys[a]][0], self.boards[keys[a]][1][b]) for a, b, _ in experience_tuple_list)
        boards[1] = ((self.boards[keys[a]][0], self.boards[keys[a]][1][c]) for a, _, c in experience_tuple_list)
        for i, e in enumerate(boards):
            e = ((convert_board(a), convert_board(b)) for a, b in e)
            boards[i] = np.array([np.concatenate(s, axis=-1) for s in e]) 
        scores[0] = np.array([self.scores[keys[a]][b] for a, b, _ in experience_tuple_list])
        scores[1] = np.array([self.scores[keys[a]][c] for a,_, c in experience_tuple_list])
        return [boards[0], scores[0], boards[1], scores[1]]
    
    def next_train(self):
        while True:
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            experience_tuple_list = self.experience_buffer_train[self.cur_train_index :
                self.cur_train_index + self.batch_sz]
            self.cur_train_index += self.batch_sz
            yield (self.gen_pairs(experience_tuple_list), np.ones(len(experience_tuple_list)))
    
    def next_val(self):
        while True:
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            experience_tuple_list = self.experience_buffer_val[self.cur_val_index :
                self.cur_val_index + self.batch_sz]
            self.cur_val_index += self.batch_sz
            yield (self.gen_pairs(experience_tuple_list), np.ones(len(experience_tuple_list)))
     
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_num_pairs(self):
        return self.samples_per_train, self.samples_per_val
        
        
        
class DataGenerator:
    """docstring for DataGenerator"""
    def __init__(self, X, boards,  num_train_frac = 0.98, batch_sz=512, load_from_file = False, 
                 save_to_file = False, maxlen = 1, file = "experience_buffer.p"):
       
        assert(not(load_from_file and save_to_file))
        self.X = X 
        self.keys = sorted(X.keys())
        if load_from_file:
            experience_buffer = pickle.load(open(file, "rb"))
            print("Experience buffer loaded from {}".format(file))
        else:
            experience_buffer = []
            for i, key in enumerate(self.keys):
                arr = range(len(X[key]))
                combinations = itertools.combinations(arr, 2)
                for pair in combinations:
                    if (pair[0] < maxlen and pair[1] < maxlen + 3):
                        experience_buffer.append((i, pair[0], pair[1]))
            experience_buffer = np.array(experience_buffer)
            print("Experience buffer generated")
        if save_to_file:
            pickle.dump(experience_buffer, open(file,"wb"))
            print("Experience buffer saved to {}".format(file))
        
        np.random.shuffle(experience_buffer)
        
        num_train = int(num_train_frac * len(experience_buffer))
        self.experience_buffer_train = experience_buffer[:num_train]
        self.experience_buffer_val = experience_buffer[num_train:]
        
        self.samples_per_train = num_train
        self.samples_per_val = len(experience_buffer) - num_train
        print("Train: {} Val: {}".format(self.samples_per_train, self.samples_per_val))
        
        self.cur_train_index = 0
        self.cur_val_index = 0
        self.batch_sz = batch_sz
    
    def gen_pairs(self, experience_tuple_list):
        
        x1 = np.array([self.X[self.keys[a]][b] for a, b, _ in experience_tuple_list])
        x2 = np.array([self.X[self.keys[a]][b] for a, _, b in experience_tuple_list])
        return [x1, x2]
    
    def next_train(self):
        while True:
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            experience_tuple_list = self.experience_buffer_train[self.cur_train_index :
                self.cur_train_index + self.batch_sz]
            self.cur_train_index += self.batch_sz
            yield (self.gen_pairs(experience_tuple_list), np.ones(len(experience_tuple_list)))
    
    def next_val(self):
        while True:
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            experience_tuple_list = self.experience_buffer_val[self.cur_val_index :
                self.cur_val_index + self.batch_sz]
            self.cur_val_index += self.batch_sz
            yield (self.gen_pairs(experience_tuple_list), np.ones(len(experience_tuple_list)))
     
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_num_pairs(self):
        return self.samples_per_train, self.samples_per_val