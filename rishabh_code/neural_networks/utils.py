import numpy as np
import tensorflow as tf
from collections import Counter
import itertools
import cPickle as pickle
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def limited_gpu_memory_session():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True 
    return tf.Session(config=config)

class DataGenerator:
    """docstring for DataGenerator"""
    def __init__(self, X, num_train_frac = 0.96, batch_sz=512, load_from_file = False, 
                 save_to_file = False, maxlen = None, file = "experience_buffer.p"):
       
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
                    if (pair[0] <= 1 and pair[1] >= 1):
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
            self.cur_train_index += self.batch_sz
            if self.cur_train_index >= self.samples_per_train:
                self.cur_train_index = 0
            experience_tuple_list = self.experience_buffer_train[self.cur_train_index :
                self.cur_train_index + self.batch_sz]
            yield (self.gen_pairs(experience_tuple_list), np.ones(len(experience_tuple_list)))
    
    def next_val(self):
        while True:
            self.cur_val_index += self.batch_sz
            if self.cur_val_index >= self.samples_per_val:
                self.cur_val_index = 0
            experience_tuple_list = self.experience_buffer_val[self.cur_val_index :
                self.cur_val_index + self.batch_sz]
            yield (self.gen_pairs(experience_tuple_list), np.ones(len(experience_tuple_list)))
     
    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_num_pairs(self):
        return self.samples_per_train, self.samples_per_val