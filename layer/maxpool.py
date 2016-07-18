import numpy as np

from neurallayer import NeuralLayer


class MaxPool(NeuralLayer):
    """
    Note: Not guaranteed to pass gradient check. This is because, while calculating the loss value with perturbations,
    the parameter perturbations themselves may cause
    """
    def __init__(self, input_dim, pool_dim):
        print("instantiating Max Pool")
        self.pool_dim = pool_dim
        output_dim = [input_dim[0]/pool_dim[0], input_dim[1]/pool_dim[1], input_dim[2]/pool_dim[2]]
        NeuralLayer.__init__(self, input_dim, output_dim, categorical_input=False)

        # Stored values for back propagation and batch updates
        self.max_binary_filter = np.zeros(self.input_dim)

    def feed_forward(self, x):
        """
        :param x: input to this neural layer, as either categorical or numerical numpy ndarray
        :return: output of this neural layer
        """
        x = x.reshape(self.input_dim)
        self.max_binary_filter = np.zeros(self.input_dim)

        output = np.zeros(self.output_dim)
        k = 0
        while k < self.input_dim[0]:
            j = 0
            while j < self.input_dim[1]:
                i = 0
                while i < self.input_dim[2]:
                    # print([k, j, i])
                    max_val = float("-inf")
                    max_index = [0, 0, 0]
                    for r in range(self.pool_dim[0]):
                        for q in range(self.pool_dim[1]):
                            for p in range(self.pool_dim[2]):
                                if x[k+r, j+q, i+p] > max_val:
                                    max_val = x[k+r, j+q, i+p]
                                    max_index = [k+r, j+q, i+p]
                    self.max_binary_filter[max_index[0], max_index[1], max_index[2]] = 1.0
                    output[k//self.pool_dim[0], j//self.pool_dim[1], i//self.pool_dim[2]] = max_val
                    i += self.pool_dim[2]
                j += self.pool_dim[1]
            k += self.pool_dim[0]
        return output

    def back_prop(self, delta):
        """
        :param delta: dL_dy back-propagated to this layer. Note: Unconventional delta notation
        :return: delta_prev
        """
        delta = delta.reshape(self.output_dim)
        delta_prev = np.zeros(self.input_dim)
        k = 0
        while k < self.input_dim[0]:
            j = 0
            while j < self.input_dim[1]:
                i = 0
                while i < self.input_dim[2]:
                    delta_prev[k, j, i] = delta[k//self.pool_dim[0], j//self.pool_dim[1], i//self.pool_dim[2]]
                    i += 1
                j += 1
            k += 1
        o = delta_prev*self.max_binary_filter
        return o

    def update(self, batch_size):
        return

    def clear_grads(self):
        return





