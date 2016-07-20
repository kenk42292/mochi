import numpy as np

from neurallayer import NeuralLayer
import utils

class VanillaRecurrent(NeuralLayer):
    def __init__(self, len_seq_in, len_elem_in, len_elem_hidden, len_seq_out, len_elem_out, eta, categorical_input=False,
                 activation=(utils.sigmoid, utils.sigmoid_prime)):
        print("Instatiating Vanilla Recurrent")
        if categorical_input:
            assert len_elem_in==1, "Categorical inputs must be represented by a sequence of identifying indeces"

        input_dim = [len_seq_in, len_elem_in]
        output_dim = [len_seq_out, len_elem_out]
        NeuralLayer.__init__(self, input_dim, output_dim, categorical_input)

        self.len_seq_in, self.len_elem_in = len_seq_in, len_elem_in
        self.len_seq_hidden, self.len_elem_hidden = self.len_seq_in+1, len_elem_hidden
        self.len_seq_out, self.len_elem_out = len_seq_out, len_elem_out

        self.Wxh = np.random.randn(self.len_elem_hidden, self.len_elem_in)/np.sqrt(self.len_elem_in)
        self.Whh = np.random.randn(self.len_elem_hidden, self.len_elem_hidden)/np.sqrt(self.len_elem_in)
        self.Why = np.random.randn(self.len_elem_out, self.len_elem_hidden)/np.sqrt(self.len_elem_in)
        self.bh = np.zeros(self.len_elem_hidden)

        self.act_fxn = activation[0]
        self.act_prime = activation[1]

        self.optimizer = optimizer

        # Stored values for back propagation and batch updates
        self.x = None
        self.z = None
        self.dL_dWxz = np.zeros_like(self.Wxz)
        self.dL_dbz = np.zeros_like(self.bz)
        self.eta = eta


    def feed_forward(self, input):
        assert len(input)==self.len_seq_in, "Input length is inconsistent with what is specified upon layer construction"

        self.x = input
        if self.categorical_input:
            hidden_states = np.zeros((self.len_seq_hidden, self.len_elem_hidden))
            outputs = np.zeros((self.len_seq_in, self.len_elem_out))
            for t in np.arange(len(self.len_seq_in)):

        else:
            return




