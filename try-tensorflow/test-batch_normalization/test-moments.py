import tensorflow as tf
import numpy as np

batch_size = 3
input_d = 4

np.random.seed(1)
nd_x = np.asarray([np.random.randint(0, 5, size=input_d) for i in range(batch_size)])

print(nd_x)

x = tf.placeholder(shape=[None, input_d], dtype=tf.float32)

mean_ten, var_ten = tf.nn.moments(x, axes=[0], keep_dims=True)

a = tf.get_variable('a', shape=[1], initializer=tf.constant_initializer(1, dtype=tf.float32))
b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer(2, dtype=tf.float32))
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=3)

print([v for v in tf.all_variables()])

with tf.Session() as sess:
    nd_x_mean, nd_x_var = sess.run((mean_ten, var_ten), feed_dict={x: nd_x})
    print(nd_x_mean)
    print(nd_x_var)