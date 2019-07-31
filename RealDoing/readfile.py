import io
import numpy as np
from numpy import float32
import tensorflow as tf
def data_iter():
    xfr = io.open(r'.\RealDoing\ph_data_600_new.txt', encoding='utf8')
    yfr = io.open(r'.\RealDoing\xvlie_data.txt', encoding='utf8')
    while True:
        x_file_line = xfr.readline()
        y_file_line = yfr.readline()
        if not(x_file_line and y_file_line):
            break
        xdatalist = x_file_line.split()
        ydatalist = y_file_line.split()
        yield (np.asarray(xdatalist, dtype=np.float32), np.asarray(ydatalist, dtype=np.int))
dataset = tf.data.Dataset.from_generator(generator=data_iter, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
with tf.name_scope('TrainingData'):
    trainDataSet = dataset.repeat(count=100).batch(20)
    trainIterator = trainDataSet.make_one_shot_iterator()
    trainBatchSize = 20
    xTrainBatch, yTrainBatch = trainIterator.get_next()
with tf.name_scope('TestData'):
    testDataSet = dataset.batch(2)
    testIterator = testDataSet.make_one_shot_iterator()
    xTestBatch, yTestBatch = testIterator.get_next()

#build model
with tf.name_scope('TrainingModelInput'):
    xInputTrainBatch = tf.reshape(tensor=xTrainBatch, shape=[trainBatchSize, -1, 12], name='xInputTrainBatch')
    yInputTrainBatch = tf.cast(x=tf.one_hot(indices=yTrainBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputTrainBatch')
    
with tf.name_scope('Model'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, None, 12], name='ModelInputX')
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(20, forget_bias=1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(20, forget_bias=1.0)
    rnnout,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, tf.unstack(x, 100, 1), dtype=tf.float32)
    rnnout = tf.stack(rnnout, 1)
    modelOut = tf.contrib.layers.fully_connected(inputs=rnnout, num_outputs=2, activation_fn=tf.nn.softmax)

with tf.name_scope('output_converter'):
    predictions = tf.argmax(modelOut, axis=1)
train_writer = tf.summary.FileWriter(logdir=".\log", graph=tf.get_default_graph())
train_writer.close()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(xTrainIterator.get_next())