
# coding: utf-8

# In[1]:


from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[2]:


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Input, Lambda, Reshape
from keras import backend as K
from keras.optimizers import SGD, Nadam
from keras import layers

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[3]:


from keras.backend.tensorflow_backend import set_session
from utils import limited_gpu_memory_session
set_session(limited_gpu_memory_session())


# In[4]:


from utils import get_available_gpus
GPUs = get_available_gpus()
print(GPUs)


# In[5]:


DATA_DIR = os.path.abspath('./')
CHECKPOINTED_WEIGHTS = os.path.join(DATA_DIR, 'checkpointed_weights_gpu.hdf5')
INIT_WEIGHTS = os.path.join(DATA_DIR, 'init_weights_base_gpu.hdf5')
EXPERIENCE_BUFFER_FILE = os.path.join(DATA_DIR, 'experience_buffer_gpu.p')


# In[6]:


from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1

def dense_relu_bn_dropout(x, size, dropout, alpha = 0.1, reg = 0):
    x = Dense(size, kernel_regularizer = l2(reg))(x)
    x = Activation('tanh')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    return x

def create_network(reg, dropout, alpha = 0.1):
    inputs = Input(shape=(INPUT_SHAPE,))
    x = dense_relu_bn_dropout(inputs, 8 , dropout, reg)
    x = dense_relu_bn_dropout(x, 4, dropout, reg)
    x = Dense(1)(x)
    base_network = Model(inputs=inputs, outputs = x)
    print(base_network.summary())
    return base_network


# In[7]:


from keras import layers

INPUT_SHAPE = 8

base_network = create_network(reg = 0.5, dropout = 0.5)
input_a = Input(shape=(INPUT_SHAPE,))
processed_a = base_network(input_a)
input_b = Input(shape=(INPUT_SHAPE,))
processed_b = base_network(input_b)
negative_b =  Lambda(lambda x: -x)(processed_b)
distance = layers.Add()([processed_a, negative_b])
out = Activation('sigmoid')(distance)
siamese_net = Model([input_a, input_b], out)
    
siamese_net.save_weights(INIT_WEIGHTS)
print(siamese_net.summary())


# In[11]:


import cPickle as pickle
MOVES = pickle.load(open("../moves_dict.p", "rb"))


# In[12]:


for key, item in MOVES.iteritems():
    MOVES[key] = np.array(item)


# In[13]:


from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
              patience=5, verbose = 1, min_lr=1e-8)
early_stopping = EarlyStopping(monitor='val_acc',
                              min_delta=1e-4,
                              patience=40,
                              verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath=CHECKPOINTED_WEIGHTS, verbose=1, save_best_only=True, monitor='val_acc')


# In[14]:


nadam = Nadam(lr=1e-3)
siamese_net.compile(optimizer=nadam, loss='binary_crossentropy', metrics=['accuracy'])
siamese_net.load_weights(INIT_WEIGHTS)


# In[16]:


from utils import DataGenerator

BATCH_SIZE = 512
load_from_file = os.path.exists(EXPERIENCE_BUFFER_FILE)
save_to_file = not load_from_file
datagen = DataGenerator(MOVES, batch_sz = BATCH_SIZE, load_from_file = load_from_file, 
                 save_to_file = save_to_file, file = EXPERIENCE_BUFFER_FILE)


# In[ ]:


NUM_TRAIN_PAIRS, NUM_VAL_PAIRS = datagen.get_num_pairs()
STEPS_PER_EPOCH = NUM_TRAIN_PAIRS//BATCH_SIZE
VALIDATION_STEPS = NUM_VAL_PAIRS//BATCH_SIZE
history = siamese_net.fit_generator(
        datagen.next_train(),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=500,
        validation_data=datagen.next_val(),
        validation_steps=VALIDATION_STEPS,
        callbacks = [reduce_lr, checkpointer, early_stopping])

