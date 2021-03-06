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

tf.set_random_seed(1)
f_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_d)
b_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_d)

f_output, f_state = tf.nn.dynamic_rnn(f_cell,
                                      inputs=inputs,
                                      sequence_length=lens,
                                      dtype=tf.float32,
                                      scope='forward_rnn')

b_output, b_state = tf.nn.dynamic_rnn(b_cell,
                                      inputs=r_inputs,
                                      sequence_length=lens,
                                      dtype=tf.float32,
                                      scope='backward_rnn')

print([v.name for v in tf.all_variables()])

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    pprint(sess.run(f_output, feed_dict={inputs: inps, lens: les})[:, 1, :])
    pprint(sess.run(b_output, feed_dict={inputs: inps, lens: les})[:, 1, :])
