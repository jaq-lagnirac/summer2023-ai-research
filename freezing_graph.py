# Justin Caringal

# Based off code from Nyla Worker on StackOverflow
# https://stackoverflow.com/questions/51826706/tensorflow-load-a-pb-file-and-then-save-it-as-a-frozen-graph-issues

'''
import tensorflow as tf

def frozen_graph_maker(export_dir,output_graph):
    tf.compat.v1.disable_eager_execution()
        
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
        output_nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            sess.graph_def,
            output_nodes# The output node names are used to select the usefull nodes
        )       
    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
        
def main():
    export_dir='./saved_models/outputs_2023-07-10-125728/pb_saved_model_2023-07-10-125728'
    output_graph = "frozen_graph.pb"
    frozen_graph_maker(export_dir,output_graph)

main()
#'''

# Based off code from Aewil on StackOverflow
# https://stackoverflow.com/questions/58119155/freezing-graph-to-pb-in-tensorflow2
#'''
import logging
import tensorflow as tf
from tensorflow.compat.v1 import graph_util
from tensorflow.python.keras import backend as K
from tensorflow import keras

# necessary !!!
tf.compat.v1.disable_eager_execution()

H5_MODEL = './saved_models/outputs_2023-07-10-125728/saved_model_2023-07-10-125728.h5'
PB_PATH = './saved_models/outputs_2023-07-10-125728/pb_saved_model_2023-07-10-125728/saved_model.pb'

h5_path = H5_MODEL
model = keras.models.load_model(h5_path)
model.summary()
# save pb
with K.get_session() as sess:
    output_names = [out.op.name for out in model.outputs]
    input_graph_def = sess.graph.as_graph_def()
    for node in input_graph_def.node:
        node.device = ""
    graph = graph_util.remove_training_nodes(input_graph_def)
    graph_frozen = graph_util.convert_variables_to_constants(sess, graph, output_names)
    tf.io.write_graph(graph_frozen, PB_PATH, as_text=False)
logging.info("save pb successfully！")
#'''