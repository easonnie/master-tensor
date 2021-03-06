import numpy as np
import tensorflow as tf

batch_size = 1
max_seq_len = 5
word_d = 4

input = tf.Variable(tf.random_uniform([batch_size, max_seq_len, word_d], 0, 10, dtype=tf.int32, seed=10), name='input')
tran_input = tf.reshape(input, [-1, 1, max_seq_len, word_d])
# batch_size

n_gram = 4
n_feature = 2

filter = tf.Variable(tf.random_uniform([1, n_gram, word_d, n_feature], 0, 5, dtype=tf.int32, seed=6), name='filter')

b = tf.Variable(tf.constant(0.5, shape=[n_feature], dtype=tf.float32), name='b')

result = tf.nn.conv2d(tf.to_float(tran_input), tf.to_float(filter), strides=[1, 1, 1, 1], padding='SAME')

tran_result = tf.nn.bias_add(tf.reshape(result, [-1, max_seq_len, n_feature]), b)

print([v.name for v in tf.all_variables()])

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print('inputs')
    print(input.eval())
    print('tran_inputs:')
    print(tran_input.eval())
    print('filters')
    print(filter.eval())
    print('result')
    print(tran_result.eval())