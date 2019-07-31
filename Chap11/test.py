import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name = "input1")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]))

output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter(logdir=".\log", graph=tf.get_default_graph())
writer.close()