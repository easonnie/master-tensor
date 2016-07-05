import tensorflow as tf
import numpy as np


line = tf.linspace(10.0, 12.0, 10, name='line')
print(line)

range = tf.cast(tf.range(10), tf.float32, name='range')
print(range)

sum = line + range

vars = tf.Variable(sum, name='var')

init_op = tf.initialize_all_variables()

# tf.set_random_seed(4)
#
# norm = tf.random_normal([2, 3], mean=1, stddev=3)
#
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(range))
    print(sess.run(line))
    print(sess.run(vars))
    save_path = saver.save(sess, './saved_checkpoints/test.ckpt')