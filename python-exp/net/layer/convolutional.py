import numpy as np
import scipy.signal as scisig

import utils
from neurallayer import NeuralLayer


class Convolutional(NeuralLayer):
    def __init__(self, input_dim, num_patterns, filter_dim, optimizer, activation=(utils.softplus, utils.softplus_prime)):
        print("instantiating Convolutional")

        self.num_patterns = num_patterns
        self.x_depth, self.x_height, self.x_width = input_dim[0], input_dim[1], input_dim[2]
        self.w_depth, self.w_height, self.w_width = filter_dim[0], filter_dim[1], filter_dim[2]
        self.z_depth, self.z_height, self.z_width \
            = self.num_patterns, self.x_height-self.w_height+1, self.x_width-self.w_width+1

        NeuralLayer.__init__(self, input_dim, [self.z_depth, self.z_height, self.z_width])

        self.Wxz = np.random.randn(self.num_patterns, self.w_depth, self.w_height, self.w_width)
        self.bz = np.zeros(self.num_patterns)
        self.act_fxn = activation[0]
        self.act_prime = activation[1]
        self.optimizer = optimizer

        # Stored values for back propagation and batch updates
        self.batch_x = None
        self.batch_z = None

    def feed_forward_single(self, x):
        return self.feed_forward(np.array([x]))[0]

    def feed_forward(self, batch_x):
        """
        :param x: input to this neural layer
        :return: output of this neural layer
        """
        batch_x = np.array([x.reshape(self.x_depth, self.x_height, self.x_width) for x in batch_x])
        self.batch_x = batch_x
        batch_z = np.array([[scisig.correlate(x, self.Wxz[k], mode="valid")
                             .reshape(self.z_height, self.z_width) + self.bz[k]
                             for k in range(self.num_patterns)] for x in batch_x])
        self.batch_z = batch_z
        return self.act_fxn(batch_z)

    def back_prop(self, deltas, update=True):
        """
        :param delta: dL_dy back-propagated to this layer. Note: Unconventional delta notation
        :return: delta_prev
        """
        avg_dL_dWxz, avg_dL_dbz, batch_dL_dx = self.get_grads(deltas)
        if update:
            self.update(avg_dL_dWxz, avg_dL_dbz)
        return batch_dL_dx

    def get_grads(self, deltas):
        batch_size = len(deltas)
        deltas = np.array([delta.reshape(self.z_depth, self.z_height, self.z_width) for delta in deltas])
        batch_dL_dz = deltas*self.act_prime(self.batch_z)
        batch_dL_dx = [0]*batch_size
        dL_dWxz = np.array([np.zeros_like(W) for W in self.Wxz])
        dL_dbz = np.zeros(self.num_patterns)
        for i in range(batch_size):
            x, dL_dz = self.batch_x[i], batch_dL_dz[i]
            for k in range(self.num_patterns):
                dL_dWxz[k] += scisig.correlate(x, dL_dz[k].reshape(1, self.z_height, self.z_width), mode="valid")
                batch_dL_dx[i] += scisig.fftconvolve(dL_dz[k].reshape(1, self.z_height, self.z_width), self.Wxz[k], mode="full")
            dL_dbz += np.sum(dL_dz, axis=(1,2))
        return dL_dWxz/batch_size, dL_dbz/batch_size, batch_dL_dx

    def update(self, dL_dWxz, dL_dbz):
        delta_w, delta_b = self.optimizer.delta(dL_dWxz, dL_dbz)
        self.Wxz -= delta_w
        self.bz -= delta_b

        """print("Resulting params in Conv: largest magnitudes are: (%f, %f) with sums of (%f, %f) and abs sums of (%f, %f)"
              % (max([np.max(self.Wxz), np.min(self.Wxz)], key=abs),
                 max([np.max(self.bz), np.min(self.bz)], key=abs),
                 np.sum(self.Wxz),
                 np.sum(self.bz),
                 np.sum(np.abs(self.Wxz)),
                 np.sum(np.abs(self.bz))))"""
