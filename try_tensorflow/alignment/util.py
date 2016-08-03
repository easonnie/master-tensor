import tensorflow as tf


def get_mask(input_len, max_time, batch_size):
    """
    :param input_len: A tensor [batch_size] specify the input length of each Batch
    :param max_time: Max time step
    :param batch_size:  Batch Size
    :return: A mask for 1d sequence inputs [batch, max_time]
    """
    lengths_transposed = tf.expand_dims(input_len, 1)
    length_tile = tf.tile(lengths_transposed, [1, max_time])

    range_ = tf.range(0, max_time, 1)
    range_row = tf.expand_dims(range_, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    mask = tf.to_float(tf.less(range_tiled, length_tile))
    return mask


def masked(inputs, input_len, batch_size=None, max_time=None):
    if max_time is None:
        max_time = int(inputs.get_shape()[1])
    if batch_size is None:
        batch_size = tf.shape(inputs)[0]
    return inputs * get_mask(input_len, max_time, batch_size)


def get_mask2d(input_len, max_time, batch_size, dimension):
    """
    :param input_len:
    :param max_time: A scalar
    :param batch_size:
    :param dimension: Dimension of each elements in the sequence
    :return: A mask for 2d sequence inputs [batch, max_time, dimension]
    """
    mask = get_mask(input_len, max_time, batch_size)
    mask_2d = tf.tile(tf.expand_dims(mask, dim=2), [1, 1, dimension])
    return mask_2d


def masked2d(inputs, input_len, batch_size=None, max_time=None, d=None):
    if d is None:
        d = int(inputs.get_shape()[2])
    if max_time is None:
        max_time = int(inputs.get_shape()[1])
    if batch_size is None:
        batch_size = tf.shape(inputs)[0]
    return inputs * get_mask2d(input_len, max_time, batch_size, d)


def _avg2d_along_time(inputs, input_len, d=None):
    """
    :param input: Input tensor [batch, max_time, dimension]
    :param input_len: Max time step for each sample [batch]
    :param d: dimension. If not provided, it'll be the last dimension of the input
    :return: Avg along time [batch, dimension]
    """
    raw_sum = tf.reduce_sum(inputs, reduction_indices=1)
    # if not d:
    #     d = int(inputs.get_shape()[2])
    # len_tile = tf.tile(tf.expand_dims(tf.to_float(input_len), 1), [1, d])
    # avg = raw_sum / len_tile
    avg = raw_sum / tf.expand_dims(tf.to_float(input_len), 1)
    return avg


def _sum2d_along_time(inputs, input_len, d=None):
    """
    :param input: Input tensor [batch, max_time, dimension]
    :param input_len: Max time step for each sample [batch]
    :param d: dimension. If not provided, it'll be the last dimension of the input
    :return: Avg along time [batch, dimension]
    """
    raw_sum = tf.reduce_sum(inputs, reduction_indices=1)
    return raw_sum


def avg2d_along_time(inputs, input_len, batch_size=None):
    d = int(inputs.get_shape()[2])
    max_time = int(inputs.get_shape()[1])
    if batch_size is None:
        batch_size = tf.shape(inputs)[0]
    return _avg2d_along_time(masked2d(inputs, input_len, batch_size), input_len)


def sum2d_along_time(inputs, input_len, batch_size=None):
    d = int(inputs.get_shape()[2])
    max_time = int(inputs.get_shape()[1])
    if batch_size is None:
        batch_size = tf.shape(inputs)[0]
    return _sum2d_along_time(masked2d(inputs, input_len, batch_size), input_len)


def last_relevant(inputs, input_len):
    batch_size = tf.shape(inputs)[0]
    max_length = int(inputs.get_shape()[1])
    output_size = int(inputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (input_len - 1)
    flat = tf.reshape(inputs, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant


def softmax_on_score(inputs, input_len):
    """
    :param inputs: [batch, time, 1]
    :param input_len: [batch]
    :return: [batch, time]
    """
    max_length = int(inputs.get_shape()[1])
    flatten_inputs = tf.reshape(inputs, [-1, max_length])
    m_softmax = masked(tf.exp(flatten_inputs), input_len)
    res_softmax = m_softmax / tf.reduce_sum(m_softmax, reduction_indices=[1], keep_dims=True)
    return res_softmax


def weighted_sum(inputs, weights):
    """
    :param inputs: [batch, max_length, dimension]
    :param weights: [batch, max_length]
    :return: [batch, dimension]
    """
    d = int(inputs.get_shape()[2])
    max_time = int(inputs.get_shape()[1])
    # flat_inputs = tf.reshape(inputs, [-1, max_time, d])
    flat_weights = tf.reshape(weights, [-1, max_time, 1])
    result = tf.reduce_sum(tf.reshape(inputs * flat_weights, [-1, max_time, d]), reduction_indices=[1])
    return result


if __name__ == '__main__':
    pass
    # from model.test.seq_1d_test_case import *
    # mask = masked(inputs=inputs, input_len=times, batch_size=batch_size)
    #
    # with tf.Session() as sess:
    #     print(mask.eval(feed_dict=feed_dict))
    #
    # from model.test.seq_2d_test_case import *
    # mask2d = avg2d_along_time(inputs=inputs, input_len=times, batch_size=batch_size)
    # last = last_relevant(inputs, times)
    #
    # with tf.Session() as sess:
    #     print(mask2d.eval(feed_dict=feed_dict))
    #     print(last.eval(feed_dict=feed_dict))