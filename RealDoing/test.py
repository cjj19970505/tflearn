import io
import numpy as np
from numpy import float32
import tensorflow as tf

t1 = tf.constant([[1,0,1,0,1], [0,0,0,0,1], [1,1,1,1,1]])
t2 = tf.constant([[1,1,1,0,1], [0,0,1,0,1], [1,1,1,1,1]])
equ = tf.equal(t1, t2)
preci =  tf.reduce_sum(tf.cast(equ, tf.float32)) / tf.reduce_sum(tf.cast(tf.ones_like(equ), tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t1