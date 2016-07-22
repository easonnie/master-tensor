import tensorflow as tf
import numpy as np


def s_layer(x, shape, scope, activate_func=None,
            output_keep_rate=1.0,
            w_init=tf.contrib.layers.xavier_initializer(),
            b_init=tf.constant_initializer(0.0, dtype=tf.float32), reuse=False):

    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        W = tf.get_variable('W', shape=shape, dtype=tf.float32,
                            initializer=w_init)
        b = tf.get_variable('b', shape=shape[1], dtype=tf.float32,
                            initializer=b_init)
        if activate_func is None:
            return tf.nn.dropout(tf.nn.xw_plus_b(x, W, b), keep_prob=output_keep_rate)
        else:
            return tf.nn.dropout(activate_func(tf.nn.xw_plus_b(x, W, b)), keep_prob=output_keep_rate)

a, b, c, d = 3, 4, 5, -150
batch = 1000
np.random.seed(1)
nd_x = np.random.randint(-10, 10, size=(batch, 3))
nd_y = [0 if a * x[0] ** 2 + b * x[1] ** 2 + d < 0 else 1 for x in nd_x[:]]
# nd_y += np.random.randint(-2, 2, batch)

print(nd_x[0])
print(nd_y[0])
# print(nd_x)
# print(nd_y)

input_x = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='input_x')
y = tf.placeholder(dtype=tf.int32, shape=[None], name='label')

out_1 = s_layer(input_x, scope='layer-1', activate_func=tf.nn.relu, shape=[3, 4],
                w_init=tf.contrib.layers.xavier_initializer(uniform=False), b_init=tf.constant_initializer(0.0))
out_2 = s_layer(out_1, scope='layer-2', activate_func=tf.nn.relu, shape=[4, 4],
                w_init=tf.contrib.layers.xavier_initializer(uniform=False), b_init=tf.constant_initializer(0.0))
out_3 = s_layer(out_2, scope='layer-3', activate_func=tf.nn.relu, shape=[4, 4],
                w_init=tf.contrib.layers.xavier_initializer(uniform=False), b_init=tf.constant_initializer(0.0))
y_o = s_layer(out_3, scope='layer-4', activate_func=None, shape=[4, 2])
y_p = tf.arg_max(y_o, dimension=1)

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(y_o, y)

train_op = tf.train.AdamOptimizer(0.001).minimize(cost)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    writer = tf.train.SummaryWriter(".", sess.graph)

    writer.flush()
    while True:
        p_cost, _ = sess.run((cost, train_op), feed_dict={input_x: nd_x, y: nd_y})
        print('prediction:', sess.run(y_p, feed_dict={input_x: [nd_x[0]]}))
        print(np.mean(p_cost))
