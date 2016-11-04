import numpy as np

from neurallayer import NeuralLayer
import utils

"""
Implements a vanilla recurrent neural net.
"""


class VanillaRecurrent(NeuralLayer):
    def __init__(self, input_dim, output_dim, optimizer,
                 activation=(utils.sigmoid, utils.sigmoid_prime)):
        print("Instatiating Vanilla Recurrent")

        NeuralLayer.__init__(self, input_dim, output_dim)

        self.Wxz = np.random.randn(output_dim[0], input_dim[0])*0.01
        self.Wyz = np.random.randn(output_dim[0], output_dim[0])*0.01
        self.bz = np.zeros((output_dim[0], 1))

        self.act_fxn = activation[0]
        self.act_prime = activation[1]

        self.optimizer = optimizer

        # Stored values for single samples: This way, can preserve state for next single sample
        # A convenience for testing (generating sentences from a single word)
        self.y = np.zeros((output_dim[0], 1))

        # Stored values for batch back propagation through time and batch updates
        self.seq_x = None
        self.seq_z = None
        self.seq_y = None

    def feed_forward(self, seq_x):
        len_seq = len(seq_x)
        self.seq_x = seq_x
        self.seq_z = np.zeros((len_seq, self.output_dim[0], 1))
        self.seq_y = np.zeros((len_seq, self.output_dim[0], 1))
        for t in np.arange(len_seq):
            self.seq_z[t] = np.dot(self.Wxz, seq_x[t]) + np.dot(self.Wyz, self.seq_y[t - 1]) + self.bz
            self.seq_y[t] = self.act_fxn(self.seq_z[t])
        return self.seq_y

    def back_prop(self, deltas, update=True):
        avg_dL_dWxz, avg_dL_dWyz, avg_dL_dbz, batch_dL_dx = self.get_grads(deltas)
        if update:
            self.update(avg_dL_dWxz, avg_dL_dWyz, avg_dL_dbz)
        return batch_dL_dx

    def get_grads(self, deltas):
        len_seq = len(deltas)
        Delta = deltas[-1]
        dL_dWxz, dL_dWyz, dL_dbz, dL_dx = 0, 0, 0, 0
        for t in range(len_seq)[::-1]:
            dL_dz = Delta * self.act_prime(self.seq_z[t])
            dL_dWxz += np.dot(dL_dz, self.seq_x[t].T)
            dL_dWyz += np.dot(dL_dz, self.seq_y[t - 1].T)
            dL_dbz += dL_dz
            dL_dx += np.dot(self.Wxz.T, dL_dz)
            Delta = deltas[t-1] + np.dot(self.Wyz.T, dL_dz)
        return dL_dWxz / len_seq, dL_dWyz / len_seq, dL_dbz / len_seq, dL_dx / len_seq

    def update(self, dL_dWxz, dL_dWyz, dL_dbz, grad_clip=5.0):
        delta_wxz, delta_wyz, delta_b = self.optimizer.delta(dL_dWxz, dL_dWyz, dL_dbz)
        self.Wxz -= np.clip(delta_wxz, -grad_clip, grad_clip)
        self.Wyz -= np.clip(delta_wyz, -grad_clip, grad_clip)
        self.bz -= np.clip(delta_b, -grad_clip, grad_clip)
