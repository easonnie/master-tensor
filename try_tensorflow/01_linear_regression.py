import tensorflow as tf
import numpy as np

raw_x = np.linspace(-1, 1, 101)
raw_y = 3 * raw_x + 4 + np.random.randn(*raw_x.shape)


def model(x, w, b):
    return tf.mul(x, w) + b

w = tf.Variable(tf.random_uniform([1], 0.0, 0.01, seed=6, name='weights'))
b = tf.Variable(tf.random_uniform([1], 0.0, 0.01, seed=6, name='bias'))

x = tf.placeholder(dtype=tf.float32, name='X')
y = tf.placeholder(dtype=tf.float32, name='Y')

cost = tf.reduce_mean(tf.squared_difference(y, model(x, w, b), name='cost'))
train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

# Try multiple Optimizer to minimize the cost

# AdamOptimizer; AdadeltaOptimizer; AdagradOptimizer

# Just the same as the numpy

# Reduce_indices is the axis in numpy
# if axis = 0, the axis is along the columns
# if axis = 1, the axis is along the row (keep_dims=True can be used to maintain the dimension)

# tf.reduce_sum(x) ==> 6
# tf.reduce_sum(x, 0) ==> [2, 2, 2]
# tf.reduce_sum(x, 1) ==> [3, 3]
# tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
# tf.reduce_sum(x, [0, 1]) ==> 6

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    init_op.run()

    for i in range(500):
        _, _w, _b, cur_cost = sess.run((train_op, w, b, cost), feed_dict={x: raw_x, y: raw_y})
        print(cur_cost)
        print(_w, _b)



