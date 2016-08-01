from try_tensorflow.gru_decomp.test_case import *
from try_tensorflow.length_mask.mask1 import *
# print([var for var in tf.all_variables()])


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
        return res + bias_term


class GRUCUSTCell:
    def __init__(self, num_units, input_size=None, activation=tf.nn.tanh):
        self._num_units = num_units
        self._activation = activation

    def zero_state(self, batch_size, dtype=tf.float32):
        return tf.zeros(shape=[batch_size, self._num_units], dtype=dtype)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(1, 2, linear([inputs, state],
                                             2 * self._num_units, True, 1.0))
                r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = self._activation(linear([inputs, r * state],
                                            self._num_units, True))
            new_h = u * state + (1 - u) * c
        return new_h, new_h, r, u


def copy_value():
    with tf.variable_scope('RNN/GRUCell/Gates/Linear', reuse=True):
        w_p = tf.get_variable(name='Matrix')

    with tf.variable_scope('RNN/GRUCUSTCell/Gates/Linear', reuse=True):
        w = tf.get_variable(name='Matrix')
        update_w = tf.assign(w, w_p)

    with tf.variable_scope('RNN/GRUCell/Gates/Linear', reuse=True):
        b_p = tf.get_variable(name='Bias')

    with tf.variable_scope('RNN/GRUCUSTCell/Gates/Linear', reuse=True):
        b = tf.get_variable(name='Bias')
        update_b = tf.assign(b, b_p)

    with tf.variable_scope('RNN/GRUCell/Candidate/Linear', reuse=True):
        cw_p = tf.get_variable(name='Matrix')

    with tf.variable_scope('RNN/GRUCUSTCell/Candidate/Linear', reuse=True):
        cw = tf.get_variable(name='Matrix')
        update_cw = tf.assign(cw, cw_p)

    with tf.variable_scope('RNN/GRUCell/Candidate/Linear', reuse=True):
        cb_p = tf.get_variable(name='Bias')

    with tf.variable_scope('RNN/GRUCUSTCell/Candidate/Linear', reuse=True):
        cb = tf.get_variable(name='Bias')
        update_cb = tf.assign(cb, cb_p)
    update = [update_w, update_b, update_cw, update_cb]
    return update

if __name__ == '__main__':

    cell = GRUCUSTCell(hidden_d)
    init_state = tf.constant(0.0, shape=[batch_size, cell.state_size], dtype=tf.float32)
    input_along_time = [inputs[:, i, :] for i in range(time)]

    mask1d = get_mask(times, max_time=time, batch_size=batch_size)
    mask2d = get_mask2d(times, max_time=time, batch_size=batch_size, dimension=hidden_d)

    output_list = []
    r_list = []
    u_list = []
    for i in range(time):
        if i == 0:
            with tf.variable_scope('RNN'):
                hidden_d_state, _, r, u = cell(input_along_time[i], init_state)
                output_list.append(hidden_d_state)
                r_list.append(r)
                u_list.append(u)
        else:
            with tf.variable_scope('RNN', reuse=True):
                hidden_d_state, _, r, u = cell(input_along_time[i], hidden_d_state)
                output_list.append(hidden_d_state)
                r_list.append(r)
                u_list.append(u)

    output_pack = tf.pack(output_list, axis=1) * mask2d

    init_op = tf.initialize_all_variables()

    update = copy_value()

    var_list = tf.all_variables()
    print([var.name for var in var_list])

    with tf.Session() as sess:
        import time
        start = time.time()
        sess.run(init_op)
        sess.run(update)
        # nd_hidden_d_state, nd_in = sess.run((hidden_d_state, input_along_time[0]), feed_dict=feed_dict)
        # print(nd_in)
        # print(nd_hidden_d_state)
        p_output_pack = sess.run(output_pack, feed_dict=feed_dict)
        print(p_output_pack)

        reset_list = sess.run(r_list, feed_dict=feed_dict)
        print(reset_list)

        end = time.time()
        print(end - start)
