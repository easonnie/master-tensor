import tensorflow as tf
import numpy as np


def get_mask(input_len, max_time, batch_size):
    lengths_transposed = tf.expand_dims(input_len, 1)
    length_tile = tf.tile(lengths_transposed, [1, max_time])

    range = tf.range(0, max_time, 1)
    range_row = tf.expand_dims(range, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    mask = tf.to_float(tf.less(range_tiled, length_tile))
    return mask


def get_mask2d(input_len, max_time, batch_size, dimension):
    mask = get_mask(input_len, max_time, batch_size)
    mask_2d = tf.tile(tf.expand_dims(mask, dim=2), [1, 1, dimension])
    return mask_2d

# input [batch, time, d]
def avg2d_along_time(input, input_len, d=None):
    raw_sum = tf.reduce_sum(input, reduction_indices=1)
    if not d:
        d = int(input.get_shape()[2])
    len_tile = tf.tile(tf.expand_dims(tf.to_float(input_len), 1), [1, d])
    avg = raw_sum / len_tile
    return avg

if __name__ == '__main__':
    batch_size = 2
    time = 5
    word_d = 4
    hidden_d = 3

    np.random.seed(8)
    tf.set_random_seed(2)

    nd_inputs = np.asarray([np.random.randint(0, 5, size=[time, word_d], dtype=np.int32) for i in range(batch_size)])
    nd_time = np.random.randint(0, 5, 2, dtype=np.int32)

    print(nd_inputs)  # [2, 5, 3]
    print(nd_time)  # [2]

    # Building Graph
    inputs = tf.placeholder(shape=[None, time, word_d], dtype=tf.float32, name='inputs')
    times = tf.placeholder(shape=[None], dtype=tf.int32, name='times')
    clear_inputs = inputs * get_mask2d(times, time, batch_size, word_d)

    feed_dict = {inputs: nd_inputs, times: nd_time}
    test_v = avg2d_along_time(clear_inputs, times)

    with tf.Session() as sess:
        print(sess.run(clear_inputs, feed_dict=feed_dict))
        print(sess.run(test_v, feed_dict=feed_dict))

    # batch_size = 4
    # max_length = 5
    # dimension = 3
    #
    # np.random.seed(2)
    # lengths_ = np.random.randint(2, 5, batch_size)
    # nd_inputs = np.asarray([np.random.randint(0, 5, size=[time, word_d], dtype=np.int32) for i in range(batch_size)])
    #
    # print(lengths_)
    #
    # input_len = tf.placeholder(tf.int32, shape=[None], name='length')
    #
    # mask = get_mask(input_len, max_time=max_length, batch_size=batch_size)
    # mask_2d = get_mask2d(input_len, max_time=max_length, batch_size=batch_size)
    # print(mask.dtype)
    #
    #
    # with tf.Session() as sess:
    #     print(sess.run(mask, feed_dict={input_len: lengths_}))
    #     print(sess.run(mask_2d, feed_dict={input_len: lengths_}))
    # print(sess.run(range_tiled))
    #
