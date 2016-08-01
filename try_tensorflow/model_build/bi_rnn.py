import tensorflow as tf
import numpy as np
from pprint import pprint

hidden_d = 3
batch = 3
length = 5
in_d = 4

np.random.seed(1)

inps = np.random.randint(5, 10, (batch, length, in_d))
les = np.random.randint(0, 5, 3)
print(inps)
print(les)

inputs = tf.placeholder(dtype=tf.float32, shape=[batch, length, in_d], name='inputs')
lens = tf.placeholder(dtype=tf.int32, shape=[batch], name='lengths')

r_inputs = tf.reverse_sequence(inputs, seq_lengths=tf.to_int64(lens), seq_dim=1)

input_list = [inputs[:, i, :] for i in range(length)]
r_input_list = [r_inputs[:, i, :] for i in range(length)]

tf.set_random_seed(1)
f_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_d)
b_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_d)


fb_output, f_state, b_state = tf.nn.bidirectional_rnn(f_cell, b_cell, inputs=input_list, dtype=tf.float32,
                                                      scope='bi_rnn')
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    pprint(sess.run(fb_output[1], feed_dict={inputs: inps, lens: les}))