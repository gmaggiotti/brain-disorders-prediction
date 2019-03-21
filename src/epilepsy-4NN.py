import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from nn_utils import read_dataset

def predict(X1):
    X1.resize((samples, neurons), refcheck=False)
    result = sess.run(l2, feed_dict={x: X1,y: test_y })
    return result[:test_y.shape[0]]

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


X,Y = read_dataset(180, 1000)
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=0.1, random_state=1)

LR = 0.001
epochs = 100
neurons = train_x.shape[1]
samples = train_x.shape[0]

x = tf.placeholder(tf.float32, shape=[None, neurons])
y = tf.placeholder(tf.float32, shape=[None, 1])
W0 = tf.Variable(tf.truncated_normal([neurons, samples], seed=5), name="W0", dtype=tf.float32)
b0 = tf.Variable(tf.truncated_normal([samples, 1]), name="bias0", dtype=tf.float32)
W1 = tf.Variable(tf.truncated_normal([samples, samples], seed=5), name="W1", dtype=tf.float32)
b1 = tf.Variable(tf.truncated_normal([samples, 1]), name="bias1", dtype=tf.float32)
W2 = tf.Variable(tf.truncated_normal([samples, 1], seed=5), name="W2", dtype=tf.float32)
b2 = tf.Variable(tf.truncated_normal([samples, 1]), name="bias2", dtype=tf.float32)
l0 = tf.sigmoid(tf.add(tf.matmul(x, W0), b0))
l1 = tf.sigmoid(tf.add(tf.matmul(l0, W1), b1))
l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)

### calculate the error
loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits=l2, labels=y))
### run the optimization
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    ###  init W & b
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        ### run the optimizer
        l2_, opt, lo = sess.run([l2,optimizer, loss], feed_dict={x: train_x, y: train_y})
        if epoch % (epochs*.1) == 0:
            error = np.mean(np.abs( train_y - l2_ ))
            test_error = np.abs((predict(test_x) - test_y))
            accuracy = 1 - np.mean(test_error)
            print 'Epoch', epoch," error:" , error , " accuracy of test-set:", accuracy

    epi_0 = np.array([[15,13,11,0,-6,-7,-5,-6,-12,-19,-25,-21,-7,6,19,24,27,28,32,35,36,40,45,48,51,50,50,48,47,46,46,44,45,42,34,22,24,26,28,27,22,17,13,16,14,14,4,-3,-10,-12,-9,-8,-9,-11,-11,-15,-17,-16,-17,-6,1,6,4,3,7,10,12,10,9,4,9,11,12,14,8,3,-1,-4,-14,-22,-29,-40,-40,-47,-48,-57,-69,-79,-79,-89,-96,-107,-115,-122,-121,-117,-103,-92,-80,-67,-59,-51,-44,-46,-45,-32,-35,-32,-31,-25,-23,-23,-25,-25,-27,-28,-27,-29,-24,-22,-16,-10,2,10,3,-3,-1,26,46,65,71,71,73,68,62,55,56,50,43,30,19,3,-3,-8,-10,-11,-15,-10,-1,7,17,24,25,30,33,31,27,23,16,11,0,-10,-23,-29,-29,-33,-36,-46,-50,-57,-51,-36,-29,-20,-12,-3,2,12]])
    epi_0 = 2 * epi_0 / float(1565.0) - 1
    print "expected output 0(No epilepsy detected), predicted output " , predict(epi_0)[0][0]

    epi_1 = np.array([[-22,-64,-121,-201,-292,-336,-398,-527,-773,-1069,-1219,-1186,-941,-661,-420,-254,-153,-96,-94,-212,-490,-762,-888,-858,-715,-521,-165,217,308,48,-366,-598,-498,-233,99,377,566,648,668,658,623,573,542,541,555,554,526,477,431,408,404,399,369,320,272,246,259,296,337,372,401,422,434,437,433,418,393,366,343,332,333,343,341,329,306,271,219,140,224,315,268,74,-284,-422,-350,-186,-56,19,4,-128,-245,-373,-469,-584,-755,-908,-981,-918,-743,-538,-362,-261,-243,-261,-314,-370,-524,-761,-906,-893,-737,-455,-199,111,113,-219,-593,-824,-720,-416,-150,151,409,544,632,684,707,657,524,351,213,181,260,370,457,507,535,549,549,544,532,523,516,518,521,519,516,512,502,491,482,475,467,461,459,454,451,449,445,435,418,397,368,333,282,208,159,290,354,183,-152,-499,-577,-415,-223,-88,-18,-61]])
    epi_1 = 2 * epi_1 / float(1565.0) - 1
    print "expected output 1(epilepsy detected), predicted output " , predict(epi_1)[0][0]

print('EOC')

