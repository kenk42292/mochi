import numpy as np


"""
An iterator that gives back random batches, each for independent processing.
Use for feedforward networks.
"""
class RandBatchIter():

    def __init__(self, data, batch_size):
        np.random.shuffle(data)
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        batch = [self.data[i] for i in
                 np.random.choice(range(len(self.data)), size=self.batch_size, replace=False)]
        batch_x = np.array([sample[0] for sample in batch])
        batch_y = np.array([sample[1] for sample in batch])
        return batch_x, batch_y




