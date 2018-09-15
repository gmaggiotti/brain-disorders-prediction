import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

size = 9245



def read_dataset():
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/dataset.csv", delimiter=";", skiprows=0)
    X = dataset[:, 0:161]
    Y = dataset[:, 161].reshape(X.__len__(), 1)
    ### applying inv sigm kernel
    #X = kernel(X)
    del dataset
    max = np.matrix(X).max()
    ### Improving gradient descent through feature scaling
    X = 2 * X / float(10) - 1

    X, Y = shuffle(X, Y)
    X = X[0:size]
    Y = Y[0:size]
    return X, Y


def predict(X1):
    X1.resize((samples, neurons), refcheck=False)
    result = sess.run(l4, feed_dict={x: X1, y: test_y, keep_prob : 1})
    return result[:test_y.shape[0]]


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


X, Y = read_dataset()
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=5)

LR = 0.0001
epochs = 4000
neurons = train_x.shape[1]
samples = train_x.shape[0]
keep_prob = tf.placeholder("float")

Xavier=0.3
x = tf.placeholder(tf.float32, shape=[None, neurons])
y = tf.placeholder(tf.float32, shape=[None, 1])
W0 = tf.Variable(tf.truncated_normal([neurons, samples], seed=1), name="W0", dtype=tf.float32) * Xavier
b0 = tf.Variable(tf.zeros([samples, 1]), name="bias0", dtype=tf.float32)
W1 = tf.Variable(tf.truncated_normal([samples, neurons], seed=0), name="W1", dtype=tf.float32) * Xavier
b1 = tf.Variable(tf.zeros([samples, 1]), name="bias1", dtype=tf.float32)
W2 = tf.Variable(tf.truncated_normal([neurons, samples], seed=0), name="W2", dtype=tf.float32) * Xavier
b2 = tf.Variable(tf.zeros([samples, 1]), name="bias2", dtype=tf.float32)
W3 = tf.Variable(tf.truncated_normal([samples, samples], seed=0), name="W3", dtype=tf.float32) * Xavier
b3 = tf.Variable(tf.zeros([samples, 1]), name="bias3", dtype=tf.float32)
W4 = tf.Variable(tf.truncated_normal([samples, 1], seed=0), name="W4", dtype=tf.float32) * Xavier
b4 = tf.Variable(tf.zeros([samples, 1]), name="bias4", dtype=tf.float32)

l0 = tf.sigmoid(tf.add(tf.matmul(x, W0), b0))
l1 = tf.sigmoid(tf.add(tf.matmul(l0, W1), b1))
l2 = tf.sigmoid(tf.add(tf.matmul(l1, W2), b2))
l3 = tf.sigmoid(tf.add(tf.matmul(l2, W3), b3))
l3_dropout = tf.nn.dropout(l3, keep_prob)
l4 = tf.sigmoid(tf.add(tf.matmul(l3_dropout, W4), b4))

### calculate the error
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l4, labels=y))
beta = 0.00001
loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(l4) )

###  decayed learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LR, global_step,
                                           100000, 0.96, staircase=True)
### apply the optimization
optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

with tf.Session() as sess:
    ###  init W & b
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        ### run the optimizer
        l4_, opt, lo = sess.run([l4, optimizer, loss], feed_dict={x: train_x, y: train_y, keep_prob : 0.5})
        #if epoch % (epochs * .01) == 0 or epoch == (epochs - 1):
        error = np.mean(np.abs(train_y - l4_))
        evar = (train_y - l4_).var()
        accuracy = 1 - np.sum(np.abs((predict(test_x) - test_y)) / samples)
        avar = ((predict(test_x) - test_y)).var()
        print  epoch, ",", error, ",", evar, ",", accuracy, ",",avar
print('EOC')


