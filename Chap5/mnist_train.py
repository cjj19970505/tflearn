#这个是有问题的
#但是把LEARNING_RATE_BASE改成0.1就没问题
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1    #把这个改成0.1就没事了艹，这个太迷了，现在0.8的话会造成loss一直上升
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./MODEL"
MODEL_NAME="model.ckpt"

def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    #train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    
    saver = tf.train.Saver()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #predict = tf.argmax(layer3, 1)

    with tf.Session() as sess:
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            train_feed = {x:xs, y_:ys}
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict=train_feed)
            if i%100 == 0:
                print("step:", step, "loss:", loss_value, "learningrate:",  sess.run(learning_rate), "crossentropymean:",sess.run(cross_entropy_mean, feed_dict=train_feed),"accuracy:",sess.run(accuracy, feed_dict=validate_feed))

mnist = input_data.read_data_sets("./database/MNIST/", one_hot=True)
train(mnist)

writer = tf.summary.FileWriter(logdir=".\log", graph=tf.get_default_graph())
writer.close()