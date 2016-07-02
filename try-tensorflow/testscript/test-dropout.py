import numpy as np
import tensorflow as tf


input = np.asarray([[1, 2, 3], [3, 4, 5]])

x = tf.placeholder(dtype=tf.float32)

r = tf.nn.dropout(x, keep_prob=0.5)

with tf.Session() as sess:

    print(sess.run(r, feed_dict={x: input}))