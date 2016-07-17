import scipy.stats as ss
import numpy as np
np.seterr(over="raise")

# Activation Functions
#COULD IT BE A PROBLEM THAT ONE FLOATING POINT ERROR IN THE ENTIRE VECTOR COULD SEND THE WHOLE VECTOR TO CATCH CLAUSE?

def sigmoid(z):
    return ss.logistic.cdf(z)


def sigmoid_prime(z):
    try:
        t = np.exp(-z)
        result = t / (1.0+t)**2.0
    except FloatingPointError:
        print("sigmoid prime overflow")
        result = sigmoid(z)*(1.0-sigmoid(z))
    return result


def softplus(z):
    try:
        result = np.log(1.0+np.exp(z))
    except FloatingPointError:
        print("softplus overflow")
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def int2Onehot(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def onehot2Int(v):
    return np.argmax(v)


