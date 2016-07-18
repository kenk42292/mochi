
import net
import optimizer
import layer
import mnist_loader
import utils
import cPickle as pickle
import numpy as np

from config import *

layers = [
    layer.Convolutional(input_dim=[1, 28, 28], num_patterns=32, filter_dim=[1, 5, 5],
                         optimizer=optimizer.RMSProp(0.01),
                         activation=(utils.softplus, utils.softplus_prime)),
    layer.MaxPool(input_dim=[32, 24, 24], pool_dim=[1, 2, 2]),
    layer.Convolutional(input_dim=[32, 12, 12], num_patterns=64, filter_dim=[32, 3, 3],
                         optimizer=optimizer.RMSProp(0.01),
                         activation=(utils.softplus, utils.softplus_prime)),
    layer.MaxPool(input_dim=[64, 10, 10], pool_dim=[1, 2, 2]),
    layer.VanillaFeedForward([64 * 5 * 5, 1], [1000, 1],
                             optimizer=optimizer.RMSProp(0.005),
                             activation=(utils.softplus, utils.softplus_prime)),
    layer.VanillaFeedForward([1000, 1], [10, 1],
                             optimizer=optimizer.RMSProp(0.005),
                             activation=(utils.softplus, utils.softplus_prime))  # 13520
]

if train_from_file:
    print("Training from File...")
    with open(layers_file, "rb") as handle:
        layers = pickle.load(handle)

neural_net = net.NeuralNet(layers, layers_file)
data_file = 'data/mnist.pkl.gz'

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
np.random.shuffle(training_data)
np.random.shuffle(validation_data)

neural_net.train(training_data, validation_data, niter=20000, batch_size=50)

# neural_net.grad_check(training_data[0][0], training_data[0][1])

