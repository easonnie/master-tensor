import tensorflow as tf
import numpy as np


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])

    # output [batch * max_length * output_size]

    index = tf.range(0, batch_size) * max_length + (length - 1)

    # [start, start + (length - 1)]

    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

# tf.shape() will compute the shape after run
# Tensor.get_shape() will return the shape by inference the definition (dtype is tensor dimension(need to be cast to int))

a = tf.Variable([[1, 2, 3], [3, 4, 5], [1, 3, 5], [2, 4, 6]], dtype=tf.float32)
b = tf.transpose(a)

x = tf.placeholder(tf.float32, [None, 4, 3])

shape_a = tf.shape(a)
shape_na = tf.shape_n([a, b])[0]

print(shape_a)
print(x.get_shape()[0])
# print(shape_na)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    a = sess.run(a)
    d_shape_a = sess.run(shape_a)
    print(d_shape_a)
    d_shape_a, d_shape_na = sess.run((shape_a, shape_na))
    print(d_shape_na)
