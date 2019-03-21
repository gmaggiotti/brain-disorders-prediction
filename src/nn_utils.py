import numpy as np
from sklearn.utils import shuffle
import os

def read_dataset(features, rows):
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../datasets/data-1k.csv", delimiter=",", skiprows=1, usecols=range(1, features)) \
        [0:rows]
    neurons = dataset.shape[1] - 1
    X = dataset[:, 0:neurons]
    Y = dataset[:, neurons].reshape(X.__len__(), 1)
    Y[Y > 1] = 0
    # Improving gradient descent through feature scaling
    # X = 2 * X / np.amax(X,0) - 1
    X = 2 * X / float(100) - 1
    return shuffle(X, Y, random_state=1)