import numpy as np

import utils
from neurallayer import NeuralLayer


class VanillaFeedForward(VanillaFeedForward):
    def __init__(self, input_dim, output_dim, optimizer, categorical_input=False):
        NeuralLayer.__init__(self, input_dim, output_dim, categorical_input)
        print("instantiating Perceptron")
        self.Wxz = np.random.randn(output_dim[0], input_dim[0]) / np.sqrt(input_dim[0])
        self.bz = np.zeros(output_dim)
        self.act_fxn = utils.linear
        self.act_prime = utils.const_one
        self.optimizer = optimizer

        # Stored values for back propagation and batch updates
        self.batch_x = None
        self.batch_z = None,