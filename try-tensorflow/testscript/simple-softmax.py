import tensorflow as tf
import numpy as np

n_x = np.asarray([[1, 2], [2, 3], [2, 3], [3, 5]])
n_y = np.asarray([1, 1, 2, 0])

x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])




