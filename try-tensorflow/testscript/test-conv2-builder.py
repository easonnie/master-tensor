import numpy as np
import tensorflow as tf


_max_seq_len = 5
_word_d = 4
batch_size = 2

np.random.seed(8)

input = np.asarray([np.random.randint(-5, 5, (_max_seq_len, _word_d)) for i in range(batch_size)])

# print(input)


g_input = tf.placeholder(dtype=tf.float32, shape=[None, _max_seq_len, _word_d])


def cnn(input: tf.Tensor, n_grams, n_features) -> tf.Tensor:
    max_seq_len = int(input.get_shape()[1])
    word_d = int(input.get_shape()[2])

    tran_input = tf.reshape(input, [-1, 1, max_seq_len, word_d])
    results = list()

    for n_gram, n_feature in zip(n_grams, n_features):
        with tf.name_scope("conv-maxpool-%s" % n_gram):
            filter = tf.to_float(tf.Variable(tf.random_uniform([1, n_gram, word_d, n_feature], 0, 5, dtype=tf.int32, seed=8)))
            f_result = tf.nn.conv2d(tran_input, filter, strides=[1, 1, 1, 1], padding='VALID')
            # batch_size * (valid_height)(1) * valid_length * n_feature

            b = tf.Variable(tf.constant(0.5, shape=[n_feature]), name="b")
            f_result = tf.nn.relu(tf.nn.bias_add(f_result, b))
            p_result = tf.reshape(tf.nn.max_pool(f_result, ksize=[1, 1, max_seq_len - n_gram + 1, 1], strides=[1, 1, 1, 1], padding='VALID'), [-1, n_feature])
            results.append(p_result)

    return tf.concat(concat_dim=1, values=results)


results = cnn(g_input, [2, 3], [4, 4])
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    print(sess.run(init_op))
    print(sess.run(results, feed_dict={g_input: input}))

# [[ 16.5   0.    5.5   0.5   0.    0.   12.5   0. ]
#  [ 23.5  21.5  10.5  20.5  13.5  17.5  21.5   5.5]]