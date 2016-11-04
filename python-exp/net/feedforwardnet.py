import cPickle as pickle
import time

import numpy as np
from data.iterator import RandBatchIter

import utils
from neuralnet import NeuralNet


class FeedForwardNet(NeuralNet):

    def __init__(self, layers, layers_file):
        print("Creating FeedForward Net")
        NeuralNet.__init__(self, layers, layers_file)

    def train(self, training_data, niter=100, batch_size=100):
        training_data_iter = RandBatchIter(training_data, batch_size)
        iter = 0
        t = time.time()
        while iter < niter:

            """ BATCH ASSIGNMENT """
            batch_x, batch_y = training_data_iter.next();

            """ TRAINING EVALUATION """
            if not (iter + 1) % 100:
                print("######################################################################################")
                print("iter: %d" % iter)
                print("time elased: " + str(time.time() - t))
                print("training loss: " + str(self.loss(batch_x, batch_y)))
                print("training validation rate %0.17f" % self.validate(batch_x, batch_y))
                with open(self.layers_file, "wb") as handle:
                    pickle.dump(self.layers, handle)
                t = time.time()

            """ TRAINING """
            activations = self.forward_pass_batch(batch_x)
            deltas = np.array([utils.softmax(activation) for activation in activations]) - batch_y
            self.backward_pass(deltas, update=True)

            """ LEARNING RATE ADJUSTMENTS """
            if not (iter + 1) % 2000:
                for layer in self.layers:
                    if hasattr(layer, "optimizer") and hasattr(layer.optimizer, "eta"):
                        layer.optimizer.eta /= 2.0

            iter += 1
        return True
