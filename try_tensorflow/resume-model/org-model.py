import tensorflow as tf
import numpy as np

batch_size = 2
time = 5
word_d = 4
hidden_d = 3

np.random.seed(8)
tf.set_random_seed(2)

nd_inputs = np.asarray([np.random.randint(0, 5, size=[time, word_d], dtype=np.int32) for i in range(batch_size)])
nd_time = np.random.randint(0, 5, batch_size, dtype=np.int32)

# nd_inputs = np.asarray([np.random.uniform(-0.02, 0.02, size=[time, word_d]) for i in range(batch_size)])
# nd_time = np.random.randint(0, 5, batch_size, dtype=np.int32)

print(nd_inputs) # [2, 5, 3]
print(nd_time) # [2]

# Building Graph
inputs = tf.placeholder(shape=[None, time, word_d], dtype=tf.float32, name='inputs')
times = tf.placeholder(shape=[None], dtype=tf.int32, name='times')

feed_dict = {inputs: nd_inputs, times: nd_time}

gru_cell = tf.nn.rnn_cell.GRUCell(hidden_d)

outputs, last_state = tf.nn.dynamic_rnn(
    gru_cell,
    inputs=inputs,
    sequence_length=times,
    dtype=tf.float32
)

tf.add_to_collection('inputs', inputs)
tf.add_to_collection('times', times)
tf.add_to_collection('outputs', outputs)

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

print(outputs.name)
print(last_state.name)

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(outputs, feed_dict=feed_dict))
    saver.save(sess, save_path='model.ckp')

# graph = tf.get_default_graph()
# graph_def = graph.as_graph_def()
