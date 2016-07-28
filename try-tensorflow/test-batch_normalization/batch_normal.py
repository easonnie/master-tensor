import tensorflow as tf
import numpy as np

import tensorflow as tf
import numpy as np

batch_size = 3
input_d = 4

np.random.seed(1)
nd_x = np.asarray([np.random.randint(0, 5, size=input_d) for i in range(batch_size)])

print(nd_x)

x = tf.placeholder(shape=[None, input_d], dtype=tf.float32)

mean, var = tf.nn.moments(x, axes=[0], keep_dims=True)

op_collection = None
f_x = tf.contrib.layers.batch_norm(x, scope='batch_norm', is_training=True, updates_collections=None, scale=True, decay=0.99)

cost = tf.reduce_sum(f_x)

train_op = tf.train.AdamOptimizer().minimize(cost)

f_y = tf.contrib.layers.batch_norm(x, scope='batch_norm', reuse=True, is_training=False, scale=True, updates_collections=None, decay=0.99)

init_op = tf.initialize_all_variables()
with tf.variable_scope('batch_norm', reuse=True):
    moving_mean = tf.get_variable(name='moving_mean')

print([var.name for var in tf.trainable_variables()])
print('-----------------')

with tf.Session() as sess:
    sess.run(init_op)
    # print(sess.run(mean, feed_dict={x: nd_x}))
    print('-----------------')
    print(sess.run((f_x, moving_mean), feed_dict={x: nd_x}))
    sess.run(train_op, feed_dict={x: nd_x})
    print(sess.run(f_y, feed_dict={x: nd_x}))

"""
[[ 2.99850106  3.99800134  0.          0.99950033]
 [ 2.99850106  0.          0.          0.99950033]
 [ 3.99800134  3.99800134  0.99950033  1.99900067]]
"""