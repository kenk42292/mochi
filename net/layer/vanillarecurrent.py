import numpy as np

from neurallayer import NeuralLayer
import utils

"""
Implements a vanilla recurrent neural net.
"""


class VanillaRecurrent(NeuralLayer):
    def __init__(self, len_seq, len_elem_in,
                 len_elem_out, optimizer,
                 activation=(utils.sigmoid, utils.sigmoid_prime)):
        print("Instatiating Vanilla Recurrent")

        self.len_seq = len_seq
        self.len_elem_in = len_elem_in
        self.len_elem_out = len_elem_out

        self.Wxz = np.random.randn(self.len_elem_out, self.len_elem_in)
        self.Wyz = np.random.randn(self.len_elem_out, self.len_elem_out)
        self.bz = np.zeros((self.len_elem_out, 1))

        self.act_fxn = activation[0]
        self.act_prime = activation[1]

        self.optimizer = optimizer

        # Stored values for single samples: This way, can preserve state for next single sample
        # A convenience for testing (generating sentences from a single word)
        self.y = np.zeros((self.len_elem_out, 1))

        # Stored values for batch back propagation through time and batch updates
        self.seq_x = None
        self.seq_z = None
        self.seq_y = None

    def feed_forward_batch(self, seq_x):
        self.seq_x = seq_x
        self.seq_z = np.zeros((self.len_seq, self.len_elem_out, 1))
        seq_outputs = np.zeros((self.len_seq + 1, self.len_elem_out, 1))
        for t in np.arange(self.len_seq):
            self.seq_z[t] = np.dot(self.Wxz, seq_x[t]) + np.dot(self.Wyz, seq_outputs[t - 1]) + self.bz
            seq_outputs[t] = self.act_fxn(self.seq_z[t])
        self.seq_y = seq_outputs
        return seq_outputs[:-1]

    def back_prop(self, deltas, update=True):
        avg_dL_dWxz, avg_dL_dWyz, avg_dL_dbz, batch_dL_dx = self.get_grads(deltas)
        if update:
            self.update(avg_dL_dWxz, avg_dL_dWyz, avg_dL_dbz)
        return batch_dL_dx

    def get_grads(self, deltas):
        dL_dz = np.zeros((self.len_elem_out, 1))
        dL_dWxz, dL_dWyz, dL_dbz, dL_dx = 0, 0, 0, 0
        for t in range(self.len_seq)[::-1]:
            Delta = deltas[t] + np.dot(self.Wyz.T, dL_dz)
            dL_dz = Delta * self.act_prime(self.seq_z[t])
            dL_dWxz += np.dot(dL_dz, self.seq_x[t].T)
            dL_dWyz += np.dot(dL_dz, self.seq_y[t - 1].T)
            dL_dbz += dL_dz
            dL_dx += np.dot(self.Wxz.T, dL_dz)
        return dL_dWxz / self.len_seq, dL_dWyz / self.len_seq, dL_dbz / self.len_seq, dL_dx / self.len_seq

    def update(self, dL_dWxz, dL_dWyz, dL_dbz):
        delta_wxz, delta_wyz, delta_b = self.optimizer.delta(dL_dWxz, dL_dWyz, dL_dbz)
        self.Wxz -= delta_wxz
        self.Wyz -= delta_wyz
        self.bz -= delta_b
