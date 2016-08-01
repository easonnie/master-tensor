import tensorflow as tf

import numpy as np


def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

# [batch * max_length]
test_pre_data = np.asarray([[1, 2, 3, 1, 3, 4],
                        [2, 3, 4, 0, 0, 0],
                        [0, 2, 1, 5, 0, 0],
                        [2, 2, 3, 4, 0, 0]], dtype=np.float32)

test_label_data = np.asarray([1, 1, 0, 2], dtype=np.int32)

test_len_data = np.asarray([6, 3, 4, 5], dtype=np.int)

test_embedding = np.random.rand(6, 3)

# print(test_embedding)

max_length = 6
# frame_size = 64
num_hidden = 5
num_classes = 3

lstm_cell = rnn_cell.BasicLSTMCell(num_hidden)

print(lstm_cell.state_size)


data = tf.placeholder(tf.int32, [None, max_length])
label = tf.placeholder(tf.int32, [None])

embedding = tf.placeholder(tf.float32, [6, 3])

# [batch * max_len * dimension]
vec_data = tf.nn.embedding_lookup(embedding, data)

lens = tf.placeholder(tf.int32, [None])

output, state = tf.nn.dynamic_rnn(
    lstm_cell,
    vec_data,
    dtype=tf.float32,
    sequence_length=lens,
)


def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_len = int(output.get_shape()[1])
    output_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_len + (length - 1)
    flat = tf.reshape(output, [-1, output_size])
    relevant = tf.gather(flat, index)
    return relevant

dropout_rate = 0.5
last = last_relevant(output, lens)
dropout_laset = tf.nn.dropout(last, keep_prob=dropout_rate)

weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
prediction = tf.argmax(tf.nn.softmax(tf.matmul(last, weight) + bias), dimension=1)

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(tf.matmul(dropout_laset, weight) + bias, label)

train_op = tf.train.AdamOptimizer(0.2).minimize(cost)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    d_output, d_state, d_last, d_prediction, d_cost = sess.run((output, state, last, prediction, cost), feed_dict={data: test_pre_data, embedding: test_embedding, lens: test_len_data, label: test_label_data})
    print('output:')
    print(d_output.shape)
    print(d_output)

    print('state:')
    print(d_state.shape)
    print(d_state)

    print('last:')
    print(d_last.shape)
    print(d_last)

    print('prediction:')
    print(d_prediction.shape)
    print(d_prediction)

    print('cost')
    print(d_cost.shape)
    print(d_cost)

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(500):
        _, cur_cost = sess.run((train_op, cost), feed_dict={data: test_pre_data, embedding: test_embedding, lens: test_len_data, label: test_label_data})
        print(i, cur_cost.sum())

    d_prediction = sess.run(prediction, feed_dict={data: test_pre_data, embedding: test_embedding, lens: test_len_data, label: test_label_data})
    print('Come up:', d_prediction)


def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    return tf.reduce_mean(cross_entropy)