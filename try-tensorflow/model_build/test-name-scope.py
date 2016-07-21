import tensorflow as tf
import numpy as np

with tf.variable_scope('outer'):
    x1 = tf.get_variable(name='x1', shape=[1, 3], dtype=tf.int32, initializer=tf.constant_initializer([1, 2, 3]))

with tf.variable_scope('outer', reuse=True):
    """
    if x2 exist then x2 will not be re-initialized !! Important
    """
    x2 = tf.get_variable(name='x1', shape=[1, 3], dtype=tf.int32, initializer=tf.constant_initializer([3, 2, 1]))

# some test code
print(x1.name)
print(x2.name)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(x2))