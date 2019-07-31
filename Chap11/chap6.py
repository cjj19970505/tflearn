import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy import float32
HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 10
TRAINING_STEPS = 1000
BATCH_SIZE = 32
TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

def generate_data(seq):
    X=[]
    y=[]
    for i in range(len(seq)-TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y,dtype=np.float32)  
def lstm_model(X, y, is_training):
    #with tf.name_scope(name='RNN'):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
    output = outputs[:, -1, :]

    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn = None, scope='fuck')
    if not is_training:
        return predictions, None, None
    loss = tf.losses.mean_squared_error(labels = y, predictions = predictions)
    train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.train.get_global_step(), optimizer="Adagrad", learning_rate=0.1)
    return predictions, loss, train_op

def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()
    X = tf.reshape(tensor=X, shape=[1, -1, 10])
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0], False)
    predictions = []
    labels = []
    Xs = []
    for i in range(TESTING_EXAMPLES):
        numX, p, l = sess.run([X, prediction, y])
        predictions.append(p)
        labels.append(l)
        Xs.append(numX)
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions-labels)**2).mean(axis=0))
    print("Mean Square Error is: %f" % rmse)

    plt.figure()
    #plt.plot(predictions, label='predictions')
    #plt.plot(labels, label='real_sin')
    plt.plot(np.reshape(a=Xs, newshape=[-1]))
    plt.legend()
    
    plt.show()

def train(sess, train_X, train_y):
    with tf.name_scope('input'):
        ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
        X, y = ds.make_one_shot_iterator().get_next()
        X = tf.reshape(tensor=X, shape=[BATCH_SIZE, -1, 10])

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)
    with tf.name_scope('summary'):
        tf.summary.scalar(name='loss', tensor=loss)
        tf.summary.histogram(name='hisLoss', values=loss)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logdir=".\log", graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        summary, _, l = sess.run([merged, train_op, loss])
        train_writer.add_summary(summary, i)
        if i % 100 == 0:
            print(str(i)+":"+str(l))
    train_writer.close()
test_start = (TRAINING_EXAMPLES + TIMESTEPS)*SAMPLE_GAP
test_end = test_start+(TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP

train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
test_X, test_Y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
with tf.Session() as sess:
    train(sess, train_X, train_y)
    #run_eval(sess, test_X, test_Y)



    