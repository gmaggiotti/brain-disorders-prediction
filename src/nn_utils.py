import numpy as np
from sklearn.utils import shuffle
import os


def enum(**enums):
    return type('Enum', (), enums)


Type = enum(epilepsy=1, tumor=2, healthy=5)


def feature_scaling(X):
    return 2 * X / np.amax(X, 0) - 1


def get_label(ds, col, label):
    return ds[ds[:, col] == label, :]


def read_dataset(features, rows, disorder_type):
    path = os.path.dirname(os.path.abspath(__file__))
    dataset = np.loadtxt(path + "/../datasets/data.csv", delimiter=",", skiprows=1, usecols=range(1, features + 1))

    disorder = get_label(dataset, features - 1, disorder_type)
    reference = get_label(dataset, features - 1, Type.healthy)
    dataset = np.concatenate((disorder, reference), axis=0)

    X = dataset[:, 0:features - 1]
    Y = dataset[:, features - 1].reshape(X.__len__(), 1)
    Y[Y == disorder_type] = 1
    Y[Y == Type.healthy] = 0
    del dataset, disorder, reference

    ### Improving gradient descent through feature scaling
    X = feature_scaling(X)
    X, Y = shuffle(X, Y, n_samples=rows)
    return X, Y
