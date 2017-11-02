import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
sys.path.insert(0, '../')

if (len(sys.argv) < 2):
    print("Please enter saved model file as argument")
    sys.exit(1)

from keras.backend.tensorflow_backend import set_session
from utils import limited_gpu_memory_session
set_session(limited_gpu_memory_session())

from keras.utils.generic_utils import get_custom_objects
from utils import CustomInitializer
get_custom_objects().update({'custom_initializer': CustomInitializer})

from keras.models import load_model
import tensorflow as tf
import os.path as osp
from keras import backend as K

DATA_DIR = '../tensorflow_model/'
input_fld = './'
weight_file = sys.argv[1]
num_output = 1
write_graph_def_ascii_flag = False
prefix_output_node_names_of_final_network = 'output_node'
output_graph_name = 'inference_graph.pb'

output_fld = DATA_DIR
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = osp.join(input_fld, weight_file)

K.set_learning_phase(0)
net_model = load_model(weight_file_path)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()

if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', osp.join(output_fld, f))


from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_fld, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: {}'.format(osp.join(output_fld, output_graph_name)))
