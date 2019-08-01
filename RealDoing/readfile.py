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
    trainBatchSize = 100
    trainDataSet = dataset.repeat().batch(trainBatchSize)
    trainIterator = trainDataSet.make_one_shot_iterator()
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
    with tf.name_scope('infer'):
        global_step = tf.Variable(initial_value=0, name='GlobalStep')
        x = tf.placeholder(dtype=tf.float32, shape=[None, None, 12], name='ModelInputX')
        
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(20, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(20, forget_bias=1.0)
        rnnout,_,_ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, tf.unstack(x, 100, 1), dtype=tf.float32)
        rnnout = tf.stack(rnnout, 1)
        modelOut = tf.contrib.layers.fully_connected(inputs=rnnout, num_outputs=2, activation_fn=None, scope='LogitsOut')
    with tf.name_scope('train'):
        y_ = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='TrainInputY')
        #loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.reshape(tensor=y_, shape=[-1,2]), logits=tf.reshape(tensor=modelOut, shape=[-1,2]), scope='loss')
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=modelOut, scope='loss')
    with tf.name_scope('output'):
        predictions = tf.argmax(modelOut, axis=-1)
    with tf.name_scope('evaluate'):
        modelOutPredict = tf.argmax(modelOut, axis=-1)
        evalSetOutput = tf.argmax(y_, axis=-1)
        precision = tf.equal(modelOutPredict, evalSetOutput)
        precision = tf.reduce_sum(tf.cast(precision, tf.float32)) / tf.reduce_sum(tf.cast(tf.ones_like(precision), tf.float32))
    with tf.name_scope('summary'):
        tf.summary.scalar(name='loss', tensor=loss)
        tf.summary.scalar(name='precision', tensor=precision)
        
    train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=global_step, optimizer="Adagrad", learning_rate=0.1)
    #train_op2 = tf.contrib.layers.optimize_loss(loss=loss2, global_step=global_step, optimizer="Adagrad", learning_rate=0.1)
    

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(logdir=".\log", graph=tf.get_default_graph())

stopTrain = False
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(xTrainIterator.get_next())
    while not stopTrain:
        xInput, yInput = sess.run([xInputTrainBatch, yInputTrainBatch])
        train_feed = {x:xInput, y_:yInput}
        summary,_,gs = sess.run([merged, train_op, global_step], feed_dict=train_feed)
        train_writer.add_summary(summary, gs)
        if gs % 1000 == 0:
            l, p = sess.run([loss, precision], feed_dict=train_feed)
            print("GlobalStep:"+str(gs), "loss:"+str(l), "Prec:"+str(p))

train_writer.close()