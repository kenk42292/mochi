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

        NeuralLayer.__init__(self, input_dim, output_dim)

        # Stored values for back propagation and updates
        self.batch_max_binary_filter = None

    def feed_forward_single(self, x):
        return self.feed_forward(np.array([x]))[0]

    def feed_forward(self, batch_x):
        """
        :param x: input to this neural layer
        :return: output of this neural layer
        """
        batch_size = len(batch_x)
        batch_x = np.array([x.reshape(self.input_dim) for x in batch_x])
        self.batch_max_binary_filter = np.array([np.zeros(self.input_dim) for _ in range(batch_size)])
        batch_output = np.array([np.zeros(self.output_dim) for _ in range(batch_size)])
        for b in range(batch_size):
            max_binary_filter = self.batch_max_binary_filter[b]
            x = batch_x[b]
            output = batch_output[b]
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
                        max_binary_filter[max_index[0], max_index[1], max_index[2]] = 1.0
                        output[k//self.pool_dim[0], j//self.pool_dim[1], i//self.pool_dim[2]] = max_val
                        i += self.pool_dim[2]
                    j += self.pool_dim[1]
                k += self.pool_dim[0]
        return batch_output

    def back_prop(self, deltas, update=False):
        """
        :param delta: dL_dy back-propagated to this layer. Note: Unconventional delta notation
        :return: delta_prev
        """
        batch_size = len(deltas)
        deltas = np.array([delta.reshape(self.output_dim) for delta in deltas])
        deltas_prev = np.array([np.zeros(self.input_dim) for delta in deltas])
        outputs = [0]*batch_size
        for b in range(batch_size):
            delta_prev = deltas_prev[b]
            delta = deltas[b]
            max_binary_filter = self.batch_max_binary_filter[b]
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
            outputs[b] = delta_prev*max_binary_filter
        return outputs

    def get_grads(self, deltas):
        return 0, 0, self.back_prop(deltas)



