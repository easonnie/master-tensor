import tensorflow as tf

batch = 3
a = tf.constant(shape=[batch, 5], value=1.0)
b = tf.constant(shape=[batch, 5], value=2.0)

a_exp = tf.expand_dims(a, dim=1)
b_exp = tf.expand_dims(b, dim=1)

ab_concat = tf.concat(1, [a_exp, b_exp])

print(ab_concat.get_shape())

with tf.Session() as sess:
    print(sess.run(ab_concat))

