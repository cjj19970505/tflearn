#对这个例子有疑惑
#w的初始选值会非常大的影响训练的速度
#比如下面stddev = 1时，正确率总是非常低
#stddev = 0.1时，正确率很快就涨高了
#这是为什么呢
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./database/MNIST/", one_hot=True)


layerInputCount = [784, 500, 10]
outputCount = 10

def showTestResult(mnist, predict, indexs):
    testFigure = plt.figure(1)
    for i in range(16):
        image = mnist.test.images[indexs[i]].reshape([28,28])
        subPlot = testFigure.add_subplot(4,4,i+1)
        
        test_feed = {x:mnist.test.images}
        testPredict = sess.run(predict, feed_dict=test_feed)
        if np.argmax(mnist.test.labels[indexs[i]]) == testPredict[indexs[i]]:
            subPlot.imshow(image)
        else:
            subPlot.imshow(image, cmap = plt.cm.gray)
        plt.title("Label:"+str(np.argmax(mnist.test.labels[indexs[i]])) + " Pred:"+str(testPredict[indexs[i]]) )
    plt.show()
        
stddev = 0.1

x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-input')
w1 = tf.Variable(tf.truncated_normal([layerInputCount[0], layerInputCount[1]], stddev=stddev))
#w1 = tf.Variable(tf.constant(0.0, shape=[layerInputCount[0], layerInputCount[1]]))
b1 = tf.Variable(tf.constant(0.1, shape=[layerInputCount[1]]))
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.truncated_normal([layerInputCount[1], layerInputCount[2]], stddev=stddev))
#w2 = tf.Variable(tf.constant(0.0, shape=[layerInputCount[1], layerInputCount[2]]))
b2 = tf.Variable(tf.constant(0.1, shape=[layerInputCount[2]]))
layer2 = tf.nn.relu(tf.matmul(layer1, w2)+b2)

w3 = tf.Variable(tf.truncated_normal([layerInputCount[2], outputCount], stddev=stddev))
#w3 = tf.Variable(tf.constant(0.0, shape=[layerInputCount[2], outputCount]))
b3 = tf.Variable(tf.constant(0.1, shape=[outputCount]))
#layer3 = tf.nn.softmax(tf.matmul(layer2, w2)+b3)
layer3 = tf.matmul(layer2, w3) + b3

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer3, labels=y_))
train_op =  tf.train.GradientDescentOptimizer(0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(layer3, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(layer3, 1)

train_step = 10000
batch_size = 500
displayInterval = 500
with tf.Session() as sess:
    validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
    test_feed = {x:mnist.test.images}
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for i in range(train_step):
        xs,ys = mnist.train.next_batch(batch_size)
        train_feed = {x:xs, y_ :ys}
        if i % displayInterval == 0:
            validate_acc = sess.run(accuracy, feed_dict=validate_feed)
            print("loss:", sess.run(loss, feed_dict={x:xs, y_ :ys}))
            print(i, validate_acc)
            print("w1:", sess.run(tf.reduce_mean(w1)), "w2:",sess.run(tf.reduce_mean(w2)), "w3:",sess.run(tf.reduce_mean(w3)))
        
        sess.run(train_op, train_feed)
        #print("loss:", sess.run(loss, feed_dict={x:xs, y_ :ys}))
    tf.train.Saver().save(sess, "./model.ckpt")