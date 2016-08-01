import tensorflow as tf
import numpy as np

batch = 2

np.random.seed(2)
np_a = np.random.randint(0, 5, size=(batch, 3))
np_b = np.random.randint(0, 5, size=(batch, 3))

print(np_a)
print(np_b)

tf.set_random_seed(4)

a = tf.placeholder(shape=[None, 3], dtype=tf.int32)
b = tf.placeholder(shape=[None, 3], dtype=tf.int32)

W = tf.get_variable(name='W', dtype=tf.int32,
                    initializer=tf.random_uniform(shape=[3, 3, 2], minval=0, maxval=5, dtype=tf.int32))

temp_1 = tf.reshape(W, [3, 3 * 2])

temp = tf.matmul(a, temp_1)
b_re = tf.expand_dims(b, dim=1)
result = tf.batch_matmul(b_re, tf.reshape(temp, [-1, 3, 2]))
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    # print(sess.run(W))
    print(sess.run((W, temp_1, temp, result), feed_dict={a: np_a, b: np_b}))