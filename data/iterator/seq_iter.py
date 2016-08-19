import numpy as np

"""
An iterator that gives back sequences of adjacent inputs.
Each input is assumed to be an index.
Use for recurrent neural nets.
"""


class SeqIter():
    def __init__(self, training_data, seq_len):
        self.seq_len = seq_len
        self.data = training_data

    def __iter__(self):
        return self

    def next(self):
        index = np.random.randint(0, len(self.data) - self.seq_len - 1)
        seq = [self.data[i] for i in range(index, index + self.seq_len + 1)]
        return np.array(seq[:-1]).reshape(self.seq_len, 1), np.array(seq[1:]).reshape(self.seq_len, 1)
