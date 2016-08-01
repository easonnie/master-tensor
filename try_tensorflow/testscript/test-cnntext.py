import numpy as np
import tensorflow as tf


max_seq_len = 5
word_d = 4
batch_size = 2

np.random.seed(8)

input = np.asarray([np.random.randint(-5, 5, (max_seq_len, word_d)) for i in range(batch_size)])

# print(input)

g_input = tf.placeholder(dtype=tf.float32, shape=[None, max_seq_len, word_d])
tran_input = tf.reshape(g_input, [-1, 1, max_seq_len, word_d])

n_gram = 2
n_feature = 4

filter_3 = tf.to_float(tf.Variable(tf.random_uniform([1, n_gram, word_d, n_feature], 0, 5, dtype=tf.int32, seed=8)))

f_result_3 = tf.nn.conv2d(tran_input, filter_3, strides=[1, 1, 1, 1], padding='VALID')
# batch_size * (valid_height)(1) * valid_length * n_feature

b_3 = tf.Variable(tf.constant(0.5, shape=[n_feature]), name="b")
f_result_3 = tf.nn.relu(tf.nn.bias_add(f_result_3, b_3))

p_result_3 = tf.reshape(tf.nn.max_pool(f_result_3, ksize=[1, 1, max_seq_len - n_gram + 1, 1], strides=[1, 1, 1, 1], padding='VALID'), [-1, n_feature])

n_gram = 3
n_feature = 4

filter_4 = tf.to_float(tf.Variable(tf.random_uniform([1, n_gram, word_d, n_feature], 0, 5, dtype=tf.int32, seed=8)))

f_result_4 = tf.nn.conv2d(tran_input, filter_4, strides=[1, 1, 1, 1], padding='VALID')
# batch_size * (valid_height)(1) * valid_length * n_feature

b_4 = tf.Variable(tf.constant(0.5, shape=[n_feature]), name="b")
f_result_4 = tf.nn.relu(tf.nn.bias_add(f_result_4, b_4))

p_result_4 = tf.reshape(tf.nn.max_pool(f_result_4, ksize=[1, 1, max_seq_len - n_gram + 1, 1], strides=[1, 1, 1, 1], padding='VALID'), [-1, n_feature])

result = tf.concat(1, [p_result_3, p_result_4])

input_shape = tf.shape(tran_input)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print('input', sess.run(tran_input, feed_dict={g_input: input}))
    print('filter_3', sess.run(filter_3))
    print('f_result_3', sess.run(f_result_3, feed_dict={g_input: input}))
    print('p_result_3', sess.run(p_result_3, feed_dict={g_input: input}))
    print('filter_3', sess.run(filter_4))
    print('f_result_4', sess.run(f_result_4, feed_dict={g_input: input}))
    print('p_result_4', sess.run(p_result_4, feed_dict={g_input: input}))
    print('result', sess.run(result, feed_dict={g_input: input}))
