"""
network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import mnist_loader

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

SCALE=1e4

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = SCALE*self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
        
    def loss(self, x, y):
        output = softmax(self.feedforward(x))
        l = -SCALE*np.sum(y * np.log(output))
        i = max(range(len(y)), key=lambda i: y[i])
        l2 = -SCALE*(np.log(output[i]))[0]
        print("l is: " + str(l))
        # print("l2 is: " + str(l2))
        return l
    
    def grad_check(self, x, y, d=1e-4, err_threshold=0.01):
        """
        Note: the grad_check CLEARS all gradients in each layer.
        Gradient check is only performed on first layer parameters.
        :param x: sample input
        :param y: desired sample output
        :param d: perturbation used in gradient calculation
        :param err_threshold: tolerance of error in gradient
        :return:
        """
        nabla_b, nabla_w = self.backprop(x, y)
        nabla_b, nabla_w = nabla_b[0], nabla_w[0]
        
        print(np.array(nabla_w).shape)

        param_names = ["dL_dWxz", "dL_dbz"]
        params = [self.weights[0], self.biases[0]]
        grads = [nabla_w, nabla_b]
        
        
        for param_name, model_param, model_grad in zip(param_names, params, grads):
            index_iter = np.nditer(model_grad, flags=["multi_index"], op_flags=["readwrite"])
            while not index_iter.finished:
                original_value = model_param[index_iter.multi_index[0]][index_iter.multi_index[1]]
                model_param[index_iter.multi_index[0]][index_iter.multi_index[1]] = original_value + d
                grad_plus = self.loss(x, y)
                model_param[index_iter.multi_index[0]][index_iter.multi_index[1]] = original_value - d
                grad_minus = self.loss(x, y)
                model_param[index_iter.multi_index[0]][index_iter.multi_index[1]] = original_value
                estimated_param_grad = (grad_plus-grad_minus)/(2*d)
                

                
                model_param_grad = model_grad[index_iter.multi_index[0]][index_iter.multi_index[1]]
                
                print(np.array(model_grad).shape)
                
                relative_error = np.abs(model_param_grad-estimated_param_grad)\
                    / (np.abs(model_param_grad)+np.abs(estimated_param_grad))
                if relative_error > err_threshold:
                    print("Gradient Check Fails for %s[%s]" % (param_name, index_iter.multi_index))
                    print("grad_plus: %0.17f" % grad_plus)
                    print("grad_minus: %0.17f" % grad_minus)
                    print("Estimated Gradient: %0.17f" % estimated_param_grad)
                    print("Model Generated Gradient: %0.17f" % model_param_grad)
                    print("relative_error: %0.17f" % relative_error)
                    print("err_threshold: %0.17f" % err_threshold)
                    print(relative_error > err_threshold)
                    return
                index_iter.iternext()
            print("Gradient check for %s passed" % param_name)
        
        
        

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x) # - np.max(x))
    return e_x / e_x.sum()
    
    
    
    
    
    
    
    
nn = Network([784, 10])
data_file = 'data/mnist.pkl.gz'
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

nn.grad_check(training_data[0][0], training_data[0][1])


