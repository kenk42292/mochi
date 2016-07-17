import numpy as np

import utils
from neurallayer import NeuralLayer


class VanillaFeedForward(NeuralLayer):
    def __init__(self, input_dim, output_dim, optimizer, categorical_input=False,
                 activation=(utils.sigmoid, utils.sigmoid_prime)):
        NeuralLayer.__init__(self, input_dim, output_dim, categorical_input)
        print("instantiating Vanilla Feed Forward")
        self.Wxz = np.random.randn(output_dim[0], input_dim[0])/np.sqrt(input_dim[0])
        self.bz = np.zeros(output_dim)
        self.act_fxn = activation[0]
        self.act_prime = activation[1]
        # Stored values for back propagation and batch updates
        self.training_vars = {
            "x": None,
            "z": None,
            "dL_dWxz": np.zeros_like(self.Wxz),
            "dL_dbz": np.zeros_like(self.bz),
        }
        self.optimizer = optimizer

    def feed_forward(self, x):
        """
        :param x: input to this neural layer, as either categorical or numerical numpy ndarray
        :return: output of this neural layer
        """
        x = x.reshape(self.input_dim)
        self.training_vars["x"] = x
        if self.categorical_input:
            # TODO: categorical input implementation
            print("categorical_input")
        else:
            self.training_vars["z"] = np.dot(self.Wxz, self.training_vars["x"])+self.bz
            return self.act_fxn(self.training_vars["z"])

    def back_prop(self, delta):
        """
        :param delta: dL_dy back-propagated to this layer. Note: Unconventional delta notation
        :return: delta_prev
        """

        # print("magnitude of sig_prime(z stored) is: " + str(np.max(np.abs(self.act_prime(self.training_vars["z"])))))

        dL_dz = delta*self.act_prime(self.training_vars["z"])
        self.training_vars["dL_dWxz"] += np.dot(dL_dz, self.training_vars["x"].T)
        self.training_vars["dL_dbz"] += dL_dz
        delta_prev = self.Wxz.T.dot(dL_dz)
        return delta_prev

    def update(self, batch_size):
        self.training_vars["dL_dWxz"] /= float(batch_size)
        self.training_vars["dL_dbz"] /= float(batch_size)
        delta_w, delta_b = self.optimizer.delta(self.training_vars["dL_dWxz"], self.training_vars["dL_dbz"])
        self.Wxz -= delta_w
        self.bz -= delta_b
        # print("(%f, %f)" % (np.sum(np.abs(delta_w)), np.sum(np.abs(delta_b))))


    def clear_grads(self):
        self.training_vars["dL_dWxz"] = np.zeros_like(self.Wxz)
        self.training_vars["dL_dbz"] = np.zeros_like(self.bz)

