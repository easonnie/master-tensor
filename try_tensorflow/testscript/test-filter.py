import numpy as np
import tensorflow as tf


input = np.random.randint(0, 10, 10)

print(input)

input_x = tf.placeholder(dtype=tf.int32, shape=[None])

length = 5

maxlen = tf.cast(tf.fill([tf.shape(input_x)[0]], length), tf.int32)
filter = tf.less_equal(input_x, maxlen)
po_filter = tf.select(filter, input_x, maxlen)

with tf.Session() as sess:
    _, d_po_filter = sess.run((filter, po_filter), feed_dict={input_x: input})
    print(d_po_filter)