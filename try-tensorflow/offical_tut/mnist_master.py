from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('../MNIST', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
_y = tf.placeholder(tf.float32, [None, 10])

# Batch size is None and set as first dimension by default

W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(- tf.mul(_y, tf.log(y)), reduction_indices=1) # [None, 1] each batch each

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init_op = tf.initialize_all_variables()

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)


with tf.Session() as sess:
    sess.run(init_op)
    for i in range(3001):
        batch = mnist.train.next_batch(256)
        sess.run(train_op, feed_dict={x: batch[0], _y: batch[1]})
        if i % 1000 == 0:
            print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, _y: mnist.test.labels}))
