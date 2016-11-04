import scipy.stats as ss
import numpy as np
np.seterr(over="raise")

# Activation Functions

def sigmoid(z):
    return ss.logistic.cdf(z)


def sigmoid_prime(z):
    try:
        t = np.exp(-z)
        result = t / (1.0+t)**2.0
    except FloatingPointError:
        result = sigmoid(z)*(1.0-sigmoid(z))
    return result


def softplus(z):
    try:
        result = np.log(1.0+np.exp(z))
    except FloatingPointError:
        result = relu(z)
    return result


def softplus_prime(z):
    return sigmoid(z)


def relu(z):
    return np.maximum(np.zeros_like(z), z)


def relu_prime(z):
    return np.piecewise(z, [z < 0.0, z >= 0.0], [0.0, 1.0])


def tanh(x):
    return np.tanh(x)


def linear(z):
    return z


def const_one(z):
    return 1

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def int2Onehot(j, v_size=10):
    e = np.zeros((v_size, 1))
    e[int(j)] = 1.0
    return e


def onehot2Int(v):
    return np.argmax(v)


