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
    f_output, f_state = tf.nn.rnn(f_cell,
                                  inputs=input_list,
                                  sequence_length=lens,
                                  dtype=tf.float32)

# b_output, b_state = tf.nn.rnn(b_cell,
#                               inputs=r_input_list,
#                               sequence_length=lens,
#                               dtype=tf.float32,
#                               scope='backward_rnn')

with tf.variable_scope('hello', reuse=True):
    fd_output, fd_state = tf.nn.dynamic_rnn(f_cell,
                                            inputs=inputs,
                                            sequence_length=lens,
                                            dtype=tf.float32)

# bd_output, bd_state = tf.nn.dynamic_rnn(b_cell,
#                                         inputs=r_inputs,
#                                         sequence_length=lens,
#                                         dtype=tf.float32,
#                                         scope='d_backward_rnn')

init_op = tf.initialize_all_variables()

# print(b_output[1].name)
# print(b_output[2].name)
# print(b_state.name)

with tf.Session() as sess:
    sess.run(init_op)
    pprint(sess.run(r_input_list, feed_dict={inputs: inps, lens: les}))
    pprint('----------------------------------------------------------')
    pprint(sess.run(r_inputs, feed_dict={inputs: inps, lens: les}))
    pprint('----------------------------------------------------------')
    pprint(sess.run(f_output, feed_dict={inputs: inps, lens: les}))
    pprint('----------------------------------------------------------')
    pprint(sess.run(fd_output, feed_dict={inputs: inps, lens: les}))
    # print(a, b)
    # print('-------------------')
    # print(c, d)

"""
[array([[[-0.93439025, -0.99959576, -0.98604   ],
        [-0.99842501, -0.99383873, -0.94187242],
        [-0.98948747, -0.99987173, -0.9786728 ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]],

       [[-0.99342197, -0.96645153, -0.99544263],
        [-0.94505644, -0.99925214, -0.80182952],
        [ 0.43272915, -0.99991304, -0.88613725],
        [ 0.60553825, -0.9996224 , -0.95348328],
        [ 0.        ,  0.        ,  0.        ]],

       [[-0.93439025, -0.99959576, -0.98604006],
        [-0.98549086, -0.99998999, -0.85404217],
        [-0.60361999, -0.99997556, -0.49899328],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]], dtype=float32),
 array([[[-0.99548864,  1.        ,  0.99999928],
        [-0.83522809,  1.        ,  0.99986213],
        [-0.97084278,  0.99999982,  0.99999118],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]],

       [[-0.83610666,  0.99999654,  0.99999177],
        [-0.64229423,  0.99999851,  0.99997491],
        [-0.51965874,  0.99999958,  0.99991411],
        [-0.86636358,  1.        ,  0.99999243],
        [ 0.        ,  0.        ,  0.        ]],

       [[-0.93736786,  0.99999607,  0.99997818],
        [-0.93526173,  1.        ,  0.99999559],
        [-0.96926057,  1.        ,  0.99999005],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]], dtype=float32)]

"""
