import numpy as np

import utils
from neurallayer import NeuralLayer


class VanillaFeedForward(NeuralLayer):
    def __init__(self, input_dim, output_dim, optimizer, categorical_input=False,
                 activation=(utils.sigmoid, utils.sigmoid_prime)):
        print("instantiating Vanilla Feed Forward")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.categorical_input = categorical_input
        self.Wxz = np.random.randn(output_dim[0], input_dim[0])/np.sqrt(input_dim[0])
        self.bz = np.zeros(output_dim)
        self.act_fxn = activation[0]
        self.act_prime = activation[1]
        self.optimizer = optimizer

        # Stored values for back propagation and batch updates
        self.batch_x = None
        self.batch_z = None,
        #self.batch_dL_dWxz = np.zeros_like(self.Wxz)
        #self.batch_dL_dbz = np.zeros_like(self.bz)

    def feed_forward(self, batch_x):
        """
        :param x: input to this neural layer, as either categorical or numerical numpy ndarray
        :return: output of this neural layer
        """
        batch_x = np.array([x.reshape(self.input_dim) for x in batch_x])
        self.batch_x = batch_x
        if self.categorical_input:
            # TODO: categorical input implementation
            print("categorical_input")
        else:
            self.batch_z = np.array([np.dot(self.Wxz, x) for x in self.batch_x]) + self.bz
            return self.act_fxn(self.batch_z)

    def back_prop(self, deltas, update=True):
        """
        :param delta: dL_dy back-propagated to this layer. Note: Unconventional delta notation
        'update': should I update during backprop?
        :return: delta_prev
        """
        batch_size = len(deltas)
        batch_dL_dz = deltas*self.act_prime(self.batch_z)
        dL_dWxz, dL_dbz = 0, 0
        for i in range(len(deltas)):
            dL_dWxz += np.dot(batch_dL_dz[i], self.batch_x[i].T)
            dL_dbz += batch_dL_dz[i]
        batch_dL_dx = np.array([np.dot(self.Wxz.T, dL_dz) for dL_dz in batch_dL_dz])
        if update:
            self.update(dL_dWxz/batch_size, dL_dbz/batch_size)
        return batch_dL_dx

    def update(self, dL_dWxz, dL_dbz):
        delta_w, delta_b = self.optimizer.delta(dL_dWxz, dL_dbz)
        self.Wxz -= delta_w
        self.bz -= delta_b
