import tensorflow as tf
import numpy as np

inputs = np.asarray([np.random.randint(0, 5, (6, 3)) for i in range(4)])
lens = np.random.randint(1, 6, 4)
# print(inputs)
print(lens)

input_x = tf.placeholder(dtype=tf.float32, shape=[None, 6, 3])
input_l = tf.placeholder(dtype=tf.int32, shape=[None])
input_xp = input_x ** 2

re_x = tf.reverse_sequence(input_xp, seq_lengths=tf.cast(input_l, dtype=tf.int64), seq_dim=1)

shape_ = tf.shape(input_xp)

with tf.Session() as sess:
    d_input_xp = sess.run(input_xp, feed_dict={input_x: inputs, input_l: lens})
    print('input:\n', d_input_xp)
    d_shape_ = sess.run(shape_, feed_dict={input_x: inputs})
    print(d_shape_)
    d_re_x = sess.run(re_x, feed_dict={input_x: inputs, input_l: lens})
    print('re_input:\n', d_re_x)