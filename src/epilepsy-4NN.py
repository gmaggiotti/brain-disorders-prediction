import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os


def feature_scaling(X):
    return 2 * X / np.amax(X, 0) - 1


def read_dataset(features, rows):
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../datasets/data.csv", delimiter=",", skiprows=1, usecols=range(1, features + 1)) \
        [0:rows]
    X = dataset[:, 0:features - 1]
    Y = dataset[:, features - 1].reshape(X.__len__(), 1)
    Y[Y > 1] = 0
    del dataset
    max = np.matrix(X).max()
    ### Improving gradient descent through feature scaling
    X = feature_scaling(X)
    X, Y = shuffle(X, Y)
    return X, Y


def predict(X1):
    X1.resize((samples, neurons), refcheck=False)
    result = sess.run(l5, feed_dict={x: X1, y: test_y, keep_prob: 1})
    return result[:test_y.shape[0]]


features = 179
rows = 5000
LR = 0.0001
epochs = 3000
Xavier = 0.8
beta = 0.0001

X, Y = read_dataset(features, rows)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=5)
neurons = train_x.shape[1]
samples = train_x.shape[0]

keep_prob = tf.placeholder("float")
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
W4 = tf.Variable(tf.truncated_normal([samples, samples], seed=0), name="W4", dtype=tf.float32) * Xavier
b4 = tf.Variable(tf.zeros([samples, 1]), name="bias4", dtype=tf.float32)
W5 = tf.Variable(tf.truncated_normal([samples, 1], seed=0), name="W5", dtype=tf.float32) * Xavier
b5 = tf.Variable(tf.zeros([samples, 1]), name="bias5", dtype=tf.float32)

l0 = tf.sigmoid(tf.add(tf.matmul(x, W0), b0))
l1 = tf.sigmoid(tf.add(tf.matmul(l0, W1), b1))
l2 = tf.sigmoid(tf.add(tf.matmul(l1, W2), b2))
l3 = tf.sigmoid(tf.add(tf.matmul(l2, W3), b3))
l4 = tf.sigmoid(tf.add(tf.matmul(l3, W4), b4))
l4_dropout = tf.nn.dropout(l4, keep_prob)
l5 = tf.sigmoid(tf.add(tf.matmul(l4_dropout, W5), b5))

### calculate the error
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=l5, labels=y))
loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(l5))

###  decayed learning rate
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LR, global_step,
                                           100000, 0.96, staircase=True)
### apply the optimization
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    ###  init W & b
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        ### run the optimizer
        l5_, opt, lo = sess.run([l5, optimizer, loss], feed_dict={x: train_x, y: train_y, keep_prob: 0.5})

        # if( epoch > 2000 ):
        #     error = np.mean(np.abs(train_y - l5_))
        #     evar = (train_y - l5_).var()
        #     accuracy = 1 - np.mean(test_error)
        #     avar = test_error.var()
        #     print  epoch, ",", error, ",", evar, ",", accuracy, ",",avar
        if epoch % (epochs * .01) == 0 or epoch == (epochs - 1):
            error = np.mean(np.abs(train_y - l5_))
            evar = (train_y - l5_).var()
            test_error = np.abs((predict(test_x) - test_y))
            accuracy = 1 - np.mean(test_error)
            avar = test_error.var()
            print("{},{},{},{},{}".format(epoch, error, evar, accuracy, avar))
    print('EOC')
