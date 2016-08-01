import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST', one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


def model(x, w, b):
    return tf.matmul(w, x) + b

# tf.matmul is matrix mul
# tf.mul is element-wise mul

x = tf.placeholder(dtype=tf.float32, shape=[784, None])
_y = tf.placeholder(dtype=tf.float32, shape=[10, None])

w = tf.Variable(tf.random_uniform(shape=[10, 784], minval=0, maxval=0.001, seed=6, dtype=tf.float32))
b = tf.Variable(tf.random_uniform(shape=[10, 1], minval=0, maxval=0.001, seed=6, dtype=tf.float32))

py_x = model(x, w, b)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, _y))

# The optimizer only support float32 cost.

train_op = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
predict_y = tf.argmax(py_x, 0)

# This is just test_variables
test_exp_result = tf.exp(py_x)
prob_sum = tf.reduce_sum(test_exp_result, reduction_indices=0)
# End test

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    init_op.run()

    for i in range(100):
        # epoch

        for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
            sess.run(train_op, feed_dict={x: trX[start:end].T, _y: trY[start:end].T})

        print(i, np.mean(np.argmax(teY, axis=1) == sess.run(predict_y, feed_dict={x: teX.T, _y: teY.T})))
