import tensorflow as tf
from try_tensorflow.alignment.util import *


class AlignCellV1(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, target=None):
        self._num_units = num_units
        self._target = target
        self.score_d = 1

    @property
    def state_size(self):
        return 0

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, _, scope=None):
        """
        :param inputs: [batch, dimension]
        :param _: some state if used
        :param scope:
        :return:
        """
        print(inputs.get_shape())
        print(target.get_shape())
        concat_state = tf.concat(1, [inputs, self._target])
        # (2, 5) [batch, dimension]
        # [batch, max_time, dimension]
        result = tf.reduce_sum(concat_state, reduction_indices=[1], keep_dims=True)
        return result, _

if __name__ == '__main__':
    from try_tensorflow.alignment.test_case import *

    target = tf.constant(1.0, dtype=tf.float32, shape=[batch_size, 1])

    outputs, state = tf.nn.dynamic_rnn(
        cell=AlignCellV1(1, target=target),
        inputs=inputs,
        sequence_length=times,
        dtype=tf.float32
    )

    # test_avg = avg2d_along_time(outputs, times)

    test = softmax_on_score(outputs, times)
    weight_s = weighted_sum(inputs, test)

    test_setup()
    with tf.Session() as sess:
        nd_out, nd_state = sess.run((outputs, state), feed_dict=feed_dict)

        print(nd_out)
        print(nd_state)
        print(test.eval(feed_dict=feed_dict))
        print(weight_s.eval(feed_dict=feed_dict))

        # print(test_avg.eval(feed_dict=feed_dict))