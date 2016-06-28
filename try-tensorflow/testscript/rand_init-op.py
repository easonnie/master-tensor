import tensorflow as tf
import numpy as np


def display(sess, src, feed_dict=None, info=None):
    print(src.name)
    print(sess.run(src, feed_dict=feed_dict))
    print(info or '')
    print('-------------')


m_a = tf.Variable(tf.cast(tf.random_uniform((3, 4), minval=-3, maxval=3, seed=2, dtype=tf.int32), dtype=tf.float32), name='m_a')
m_b = tf.Variable(tf.cast(tf.random_uniform((4, 3), minval=0, maxval=5, seed=3, dtype=tf.int32), dtype=tf.float32), name='m_b')

# Test for argmax of matrix
max_a_0 = tf.argmax(m_a, name='max_a_0', dimension=0)
max_a_1 = tf.argmax(m_a, name='max_a_1', dimension=1)
#

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    display(sess, m_a)
    display(sess, m_b)

    display(sess, max_a_0)
    display(sess, max_a_1)