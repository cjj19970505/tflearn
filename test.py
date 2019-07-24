import tensorflow as tf
from numpy.random import RandomState

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
    
with tf.variable_scope("foo",reuse=True):
    v = tf.get_variable("v")