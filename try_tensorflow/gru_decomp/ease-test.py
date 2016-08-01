import tensorflow as tf

tf.set_random_seed(2)
w = tf.get_variable("Matrix", [4, 3], dtype=tf.float32)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(w))