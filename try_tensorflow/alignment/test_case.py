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

# nd_inputs = np.asarray([np.random.uniform(-0.02, 0.02, size=[time, word_d]) for i in range(batch_size)])
# nd_time = np.random.randint(0, 5, batch_size, dtype=np.int32)

print(nd_inputs) # [2, 5, 3]
print(nd_time) # [2]

# Building Graph
inputs = tf.placeholder(shape=[None, time, word_d], dtype=tf.float32, name='inputs')
times = tf.placeholder(shape=[None], dtype=tf.int32, name='times')

feed_dict = {inputs: nd_inputs, times: nd_time}
init_op = None


def test_setup():
    global init_op
    init_op = tf.initialize_all_variables()
    var_list = tf.all_variables()
    print([var.name for var in var_list])

if __name__ == '__main__':

    test_setup()

    with tf.Session() as sess:
        import time
        sess.run(init_op)
        start = time.time()
        



        end = time.time()
        print(end - start)

'''
[[[-0.1711551  -0.47216749  0.04282239]
  [-0.33255923 -0.0455765  -0.05557477]
  [-0.         -0.          0.        ]
  [-0.         -0.          0.        ]
  [-0.         -0.          0.        ]]

 [[-0.63924587 -0.29723939  0.05378766]
  [-0.74984443  0.21297073  0.19110563]
  [-0.90573573  0.32340288  0.37086886]
  [-0.9335134  -0.67408186  0.35778382]
  [-0.         -0.          0.        ]]]
       '''