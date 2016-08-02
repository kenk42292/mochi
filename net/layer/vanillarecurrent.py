import numpy as np

from neurallayer import NeuralLayer
import utils

"""
Implements a vanilla recurrent neural net.
"""


class VanillaRecurrent(NeuralLayer):
    def __init__(self, len_seq_in, len_elem_in,
                 len_elem_hidden,
                 len_seq_out, len_elem_out,
                 optimizer, categorical_input=False,
                 activation_hidden=(utils.sigmoid, utils.sigmoid_prime),
                 activation_out=(utils.softplus, utils.softplus_prime)):
        print("Instatiating Vanilla Recurrent")
        if categorical_input:
            assert len_elem_in == 1, "Categorical inputs must be represented by a sequence of identifying indeces"

        self.len_seq_in, self.len_elem_in = len_seq_in, len_elem_in
        self.len_seq_hidden, self.len_elem_hidden = self.len_seq_in + 1, len_elem_hidden
        self.len_seq_out, self.len_elem_out = len_seq_out, len_elem_in

        self.Wxh = np.random.randn(self.len_elem_hidden, self.len_elem_in) / np.sqrt(self.len_elem_in)
        self.Whh = np.random.randn(self.len_elem_hidden, self.len_elem_hidden) / np.sqrt(self.len_elem_in)
        self.Why = np.random.randn(self.len_elem_out, self.len_elem_hidden) / np.sqrt(self.len_elem_in)
        self.bh = np.zeros(self.len_elem_hidden)

        self.act_hidden_fxn = activation_hidden[0]
        self.act_hidden_prime = activation_hidden[1]

        self.act_out_fxn = activation_out[0]
        self.act_out_prime = activation_out[1]

        self.optimizer = optimizer

        # Stored values for back propagation and batch updates
        self.x = None
        self.z = None


    def feed_forward(self, batch_x):
        self.x = batch_x
        if self.categorical_input:
            hidden_states = np.zeros((self.len_seq_hidden, self.len_elem_hidden))
            outputs = np.zeros((self.len_seq_in, self.len_elem_out))
            for t in np.arange(len(self.len_seq_in)):
        else:
            batch_y = []
            for x in batch_x:
                hidden_states = np.zeros((self.len_seq_in, self.len_elem_hidden))
                for i in range(self.len_seq_in):
                    hidden_states[i] = self.act_hidden_fxn(np.dot(self.Wxh, x) + self.bh)
