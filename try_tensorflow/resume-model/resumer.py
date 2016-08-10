import tensorflow as tf
import numpy as np

batch_size = 2
time = 5
word_d = 4
hidden_d = 3

np.random.seed(8)
tf.set_random_seed(2)

nd_inputs = np.asarray([np.random.randint(0, 5, size=[time, word_d], dtype=np.int32) for i in range(batch_size)])
nd_time = np.random.randint(0, 5, batch_size, dtype=np.int32)

saver = tf.train.import_meta_graph('model.ckp.meta')

with tf.Session() as sess:
    saver.restore(sess, save_path='model.ckp')
    inputs = tf.get_collection('inputs')[0]
    times = tf.get_collection('times')[0]
    outputs = tf.get_collection('outputs')[0]

    feed_dict = {inputs: nd_inputs, times: nd_time}
    print(sess.run(outputs, feed_dict=feed_dict))