import numpy as np


"""
An iterator that gives back sequences of adjacent inputs.
Use for recurrent neural nets.
"""
class SeqIter():

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __iter__(self):
        return self

    def next(self):
        index = np.random.randint(0, len(self.data)-self.seq_len)
        seq = [self.data[i] for i in range(index, index+self.seq_len)]
        seq_x = np.array([sample[0] for sample in seq])
        seq_y = np.array([sample[1] for sample in seq])
        return seq_x, seq_y