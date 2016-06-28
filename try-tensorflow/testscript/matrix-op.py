import tensorflow as tf
import numpy as np

np_x = np.asmatrix([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
np_y = np.asmatrix([[3, 4, 5], [1, 3, 5], [2, 4, 6]])

'''

x
1 2 3
2 3 4
3 4 5

y
3 4 5
1 3 5
2 4 6

'''

tf_x = tf.Variable(np_x, dtype=tf.float32)
tf_y = tf.Variable(np_y, dtype=tf.float32)

tf_v = tf.Variable([[1], [2], [3], [4]], dtype=tf.float32)
tf_vT = tf.Variable([1, 2, 3, 4], dtype=tf.float32)

# Some test value for matrix
el_xy = tf.mul(tf_x, tf_y)
ma_xy = tf.matmul(tf_x, tf_y)

reduce_sum1 = tf.reduce_sum(tf_y)
reduce_sum2 = tf.reduce_sum(tf_y, reduction_indices=0)
reduce_sum3 = tf.reduce_sum(tf_y, reduction_indices=1, keep_dims=True)
#

# Test for tf.argsort and tf.argmax
max_v = tf.argmax(tf_v, dimension=0)
max_vT = tf.argmax(tf_v, dimension=0)
#

init_op = tf.initialize_all_variables()


def display(sess, src, feed_dict=None, name=None, info=None):
    print(' ', name, ':')
    print(sess.run(src, feed_dict=feed_dict))
    print(info or '')
    print('-------------')

with tf.Session() as sess:
    sess.run(init_op)
    display(sess, tf_x, name='x')
    display(sess, tf_y, name='y')
    display(sess, el_xy, name='el_xy')
    display(sess, ma_xy, name='ma_xy')

    display(sess, reduce_sum1, name='reduce_sum1', info='reduce sum of matrix-y (no given reduction_indices)')
    display(sess, reduce_sum2, name='reduce_sum2', info='reduce sum of matrix-y (reduction_indices = 0)')
    display(sess, reduce_sum3, name='reduce_sum3', info='reduce sum of matrix-y (reduction_indices = 1)')

    display(sess, tf_v, name='tf_v')
    display(sess, tf_vT, name='tf_vT')

    display(sess, max_v, name='max_v')
    display(sess, max_vT, name='max_vT')