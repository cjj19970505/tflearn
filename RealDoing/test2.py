import io
import numpy as np
from numpy import float32
import tensorflow as tf
import matplotlib.pyplot as plt
def train_data_iter():
    xfr = io.open(r'.\RealDoing\train_ph.txt', encoding='utf8')
    yfr = io.open(r'.\RealDoing\train_bin.txt', encoding='utf8')
    while True:
        x_file_line = xfr.readline()
        y_file_line = yfr.readline()
        if not(x_file_line and y_file_line):
            break
        xdatalist = x_file_line.split()
        ydatalist = y_file_line.split()
        yield (np.asarray(xdatalist, dtype=np.float32), np.asarray(ydatalist, dtype=np.int))

def test_data_iter():
    xfr = io.open(r'.\RealDoing\test_ph.txt', encoding='utf8')
    yfr = io.open(r'.\RealDoing\test_bin.txt', encoding='utf8')
    while True:
        x_file_line = xfr.readline()
        y_file_line = yfr.readline()
        if not(x_file_line and y_file_line):
            break
        xdatalist = x_file_line.split()
        ydatalist = y_file_line.split()
        yield (np.asarray(xdatalist, dtype=np.float32), np.asarray(ydatalist, dtype=np.int))

def display_data_iter():
    xfr = io.open(r'.\RealDoing\all_ph.txt', encoding='utf8')
    yfr = io.open(r'.\RealDoing\all_bin.txt', encoding='utf8')
    while True:
        x_file_line = xfr.readline()
        y_file_line = yfr.readline()
        if not(x_file_line and y_file_line):
            break
        xdatalist = x_file_line.split()
        ydatalist = y_file_line.split()
        yield (np.asarray(xdatalist, dtype=np.float32), np.asarray(ydatalist, dtype=np.int))

train_dataset = tf.data.Dataset.from_generator(generator=train_data_iter, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
test_dataset = tf.data.Dataset.from_generator(generator=test_data_iter, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
display_dataset = tf.data.Dataset.from_generator(generator=display_data_iter, output_types=(tf.float32, tf.int32), output_shapes=(tf.TensorShape(None), tf.TensorShape(None)))
with tf.name_scope('TrainingData'):
    trainBatchSize = 100
    trainDataSet = train_dataset.repeat().batch(trainBatchSize)
    trainIterator = trainDataSet.make_one_shot_iterator()
    xTrainBatch, yTrainBatch = trainIterator.get_next()
with tf.name_scope('TestData'):
    testBatchSize = 50
    testDataSet = test_dataset.repeat().batch(testBatchSize)
    testIterator = testDataSet.make_one_shot_iterator()
    xTestBatch, yTestBatch = testIterator.get_next()
with tf.name_scope('displayData'):
    displayBatchSize = 2
    displayDataSet = display_dataset.repeat().batch(displayBatchSize)
    displayIterator = displayDataSet.make_one_shot_iterator()
    xDisplayBatch, yDisplayBatch = displayIterator.get_next()

#build model
with tf.name_scope('TrainingModelInput'):
    xInputTrainBatch = tf.reshape(tensor=xTrainBatch, shape=[trainBatchSize, -1, 12], name='xInputTrainBatch')
    yInputTrainBatch = tf.cast(x=tf.one_hot(indices=yTrainBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputTrainBatch')

with tf.name_scope('TestModelInput'):
    xInputTestBatch = tf.reshape(tensor=xTestBatch, shape=[testBatchSize, -1, 12], name='xInputTestBatch')
    yInputTestBatch = tf.cast(x=tf.one_hot(indices=yTestBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputTestBatch')
with tf.name_scope('DisplayModelInput'):
    xInputDisplayBatch = tf.reshape(tensor=xDisplayBatch, shape=[displayBatchSize, -1, 12], name='xInputDisplayBatch')
    yInputDisplayBatch = tf.cast(x=tf.one_hot(indices=yDisplayBatch, depth=2, on_value=1, off_value=0, dtype=tf.int32), dtype=tf.float32, name='yInputDisplayBatch')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x, y = sess.run([xInputDisplayBatch, yInputDisplayBatch])
    
    phs = x.reshape([displayBatchSize,-1])
    bins = np.argmax(y, axis=2)
    t = range(phs.shape[1])

    for i in range(phs.shape[0]):
        figure = plt.figure()
        plt.plot(t, phs[i])
        pltRange = np.max(phs[i])-np.min(phs[i])
        pltBase = np.min(phs[i])
        plt.plot(t, pltRange/2*np.repeat(bins[i], 12)+pltBase)
        plt.plot(t, pltRange/2*np.repeat(bins[i], 12)+pltBase+pltRange/2)
    plt.show()
