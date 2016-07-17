import numpy as np
import scipy.signal as scisig

import utils
from neurallayer import NeuralLayer


class Convolutional2(NeuralLayer):
    def __init__(self, input_dim, num_patterns, filter_dim, optimizer, activation=(utils.softplus, utils.softplus_prime)):
        print("instantiating Convolutional2")
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
        # Stored values for back propagation and batch updates
        self.training_vars = {
            "x": None,
            "z": None,
            "dL_dWxz": np.array([np.zeros_like(W) for W in self.Wxz]),
            "dL_dbz": np.zeros(self.num_patterns),
        }
        self.optimizer = optimizer

    def feed_forward(self, x):
        """
        :param x: input to this neural layer, as either categorical or numerical numpy ndarray
        :return: output of this neural layer
        """
        x = x.reshape(self.x_depth, self.x_height, self.x_width)
        self.training_vars["x"] = x
        z = np.array([scisig.correlate(x, self.Wxz[k], mode="valid")
                     .reshape(self.z_height, self.z_width) + self.bz[k]
                      for k in range(self.num_patterns)])
        self.training_vars["z"] = z
        return self.act_fxn(z)

    def back_prop(self, delta):
        """
        :param delta: dL_dy back-propagated to this layer. Note: Unconventional delta notation
        :return: delta_prev
        """
        delta = delta.reshape(self.z_depth, self.z_height, self.z_width)
        dL_dz = delta*self.act_prime(self.training_vars["z"])
        dL_dx = np.zeros_like(self.training_vars["x"])

        # print("delta sum: " + str(np.sum(np.abs(delta))))

        # TODO: Optimize by storing flipped x before for-loop
        x_flipped = self.training_vars["x"]
        for k in range(self.num_patterns):
            self.training_vars["dL_dWxz"][k] \
                += scisig.correlate(x_flipped, dL_dz[k].reshape(1, self.z_height, self.z_width), mode="valid")
            self.training_vars["dL_dbz"][k] += np.sum(dL_dz[k])
            dL_dx += scisig.fftconvolve(dL_dz[k].reshape(1, self.z_height, self.z_width), self.Wxz[k], mode="full")
        return dL_dx

    def update(self, batch_size):
        self.training_vars["dL_dWxz"] /= float(batch_size)
        self.training_vars["dL_dbz"] /= float(batch_size)
        delta_w, delta_b = self.optimizer.delta(self.training_vars["dL_dWxz"], self.training_vars["dL_dbz"])
        self.Wxz -= delta_w
        self.bz -= delta_b
        # print("(%f, %f)"%(np.sum(np.abs(delta_w)), np.sum(np.abs(delta_b))))

    def clear_grads(self):
        self.training_vars["dL_dWxz"] = np.array([np.zeros_like(W) for W in self.Wxz])
        self.training_vars["dL_dbz"] = np.zeros(self.num_patterns)
