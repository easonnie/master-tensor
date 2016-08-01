import tensorflow as tf
import numpy as np


def body(i, x, x_):
    a = tf.zeros(shape=[1, 1], dtype=tf.int32)
    b = tf.constant(value=1, shape=[1, 1], dtype=tf.int32)
    c = a + b
    return i + 1, a, b


def condition(i, x, x_):
    return tf.less(i, tf.constant(10))

x = tf.constant(1, shape=[1, 1])
x_ = tf.constant(1, shape=[1, 1])
i = tf.constant(0)
# vars = tf.tuple([x, x])
var = tf.tuple([i, x, x_])

result = tf.while_loop(condition, body, var)
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(result[0]))

'''
sure that body() is a callable taking a list of tensors and returning a list of tensors of the same length and with the same types as the input. That's how While_loop works.
Each returns are send back as input argument. That is, previous return are input parameter of next iteration.
'''