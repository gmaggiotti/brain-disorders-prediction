import numpy as np
from nn_utils import read_dataset

size = 1000


def sigmoid(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


X, y = read_dataset(180, 1000)

np.random.seed(5)
# synapses
w0 = 2 * np.random.random((X.size / X.__len__(), X.__len__())) - 1
w1 = 2 * np.random.random((X.__len__(), X.__len__())) - 1
w2 = 2 * np.random.random((X.__len__(), 1)) - 1

# training step
for j in xrange(60000):

    # Calculate forward through the network.
    l0 = X
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))
    l3 = sigmoid(np.dot(l2, w2))

    # Error back propagation of errors using the chain rule.
    l3_error = y - l3
    if (j % 10) == 0:
        print("Error: " + str(np.mean(np.abs(l3_error))))

    l3_adjustment = l3_error * sigmoid(l3, deriv=True)  # (y-a).d/dw(-a), a = sigmoid(Sum Xi*Wi)
    l2_error = l3_adjustment.dot(w2.T)

    l2_adjustment = l2_error * sigmoid(l2, deriv=True)  # (y-a).d/dw(-a), a = sigmoid(Sum Xi*Wi)
    l1_error = l2_adjustment.dot(w1.T)

    l1_adjustment = l1_error * sigmoid(l1, deriv=True)  # (y-a).d/dw(-a), a = sigmoid(Sum Xi*Wi)

    # update weights for all the synapses (no learning rate term)
    w2 += l2.T.dot(l3_adjustment)
    w1 += l1.T.dot(l2_adjustment)
    w0 += l0.T.dot(l1_adjustment)

print("Output after training")
print(l3)


def predict(X1):
    l0 = np.zeros((X.__len__(), X.size / X.__len__()))
    max = np.matrix(X1).max()
    l0[0] = 2 * np.asanyarray(X1, dtype=np.float32) / max - 1
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))
    l3 = sigmoid(np.dot(l2, w2))
    return l3[0][0]  # since process X1[0] output would be l2[0]


test_dataset = [1, 9, 19, 33, 16, 2, 1]
result = predict(test_dataset)
print("expected output 1, predicted output " + repr(result))
assert (result > 0.95), "Test Failed. Exepected result > 0.95"

test_dataset = [1, 0, 1, 4, 1, 3, 1]
result = predict(test_dataset)
print("expected output 0, predicted output " + repr(result))
assert (result < 0.95), "Test Failed. Exepected result < 0.95"
