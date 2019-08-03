import io
import numpy as np
from numpy import float32
import tensorflow as tf
import matplotlib.pyplot as plt
y_ = tf.constant(value=[[[0,1], [0,1], [1, 0]], [[1, 0], [0, 1], [1, 0]]], dtype=tf.float32)
v1 = tf.constant(value=[[[0.1,0.9], [0.2,0.8], [0.3, 0.7]], [[0.4, 0.6], [0.11, 0.89], [0.56, 0.44]]])
v2 = tf.unstack(v1, axis=-1)
v2 = tf.stack([v2[1], v2[0]], axis=-1)
dotResult = tf.reduce_sum(v2*y_, axis=-1)
distribution = tf.reduce_sum(dotResult, axis=0)
distribution = distribution / tf.reduce_sum(distribution)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    