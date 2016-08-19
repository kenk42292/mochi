import numpy as np
import utils
from seq_iter import SeqIter

"""
An iterator that gives back sequences of adjacent inputs.
Unlike its parent class, each input is assumed to be a categorical index. Thus, this iterator
converts categorical indeces to one-hot vectors.
Use for recurrent neural nets.
"""


class CatSeqIter(SeqIter):
    def __init__(self, training_data, seq_len, elem_len):
        SeqIter.__init__(self, training_data, seq_len)
        self.elem_len = elem_len

    def next(self):
        categorical_seq_in, categorical_seq_out = SeqIter.next(self)
        return np.array([utils.int2Onehot(i, self.elem_len) for i in categorical_seq_in]), np.array(
            [utils.int2Onehot(i, self.elem_len) for i in categorical_seq_out])
