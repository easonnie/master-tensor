import tensorflow as tf
import numpy as np
from pprint import pprint

hidden_d = 3
batch = 1
length = 5
in_d = 4

np.random.seed(1)

inps = np.random.randint(5, 10, (batch, length, in_d))
les = np.random.randint(0, 5, 1)
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


with tf.variable_scope('hello'):
    out1, *__ = tf.nn.bidirectional_rnn(f_cell, b_cell, input_list, dtype=tf.float32, sequence_length=lens, scope="ss")
    # hidden ss_FW scope for froward
    # hidden ss_BW scope for backward

with tf.variable_scope("hello", reuse=True):
    out2_f, *__ = tf.nn.dynamic_rnn(f_cell, inputs, dtype=tf.float32, sequence_length=lens, scope="ss_FW")
    out2_b, *__ = tf.nn.dynamic_rnn(b_cell, r_inputs, dtype=tf.float32, sequence_length=lens, scope="ss_BW")

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    pprint(sess.run(out1, feed_dict={inputs: inps, lens: les}))
    pprint(sess.run((out2_f, out2_b), feed_dict={inputs: inps, lens: les}))
