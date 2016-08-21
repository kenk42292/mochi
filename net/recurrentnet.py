import numpy as np

import utils
from data.iterator import CatSeqIter
from neuralnet import NeuralNet
import time
import cPickle as pickle


class RecurrentNet(NeuralNet):
    """
    A Neural Network that supports Recurrent layers.
    The only difference between a feedforward net and a recurrent net is that, in a recurrent net,
    the mini-batches are adjacent, and in order. In other words, they are SEQUENCES.
    Of course, using the feedforward mini-batches as sequences takes away the ability of
    a recurrent net to do mini-batch gradient descent with batches of sequences.
    But that's alright - in fact, that's usually how it's done.
    """

    def __init__(self, layers, layers_file, len_seq, domain_size, index2word):
        print("Creating Recurrent Net")
        NeuralNet.__init__(self, layers, layers_file)
        self.len_seq = len_seq
        self.len_elem = domain_size
        print('$$$$$$$$$$$$$$$$$')
        print(self.len_seq)
        print(self.len_elem)
        self.index2Word = index2word

    def train(self, training_data, niter=100, len_seq=20):
        training_data_iter = CatSeqIter(training_data, len_seq, self.len_elem)

        iter = 0
        t = time.time()
        while iter < niter:

            """ SEQUENCE ASSIGNMENT """
            seq_x, seq_y = training_data_iter.next()
            """ TRAINING EVALUATION """
            if not (iter + 1) % 1000 or not iter:
                print("######################################################################################")
                print("iter: %d" % iter)
                print("time elased: " + str(time.time() - t))
                print("training loss: " + str(self.loss(seq_x, seq_y)))
                self.validate()
                with open(self.layers_file, "wb") as handle:
                    pickle.dump(self.layers, handle)
                t = time.time()

            """ TRAINING """
            activations = self.forward_pass_batch(seq_x)
            deltas = np.array([utils.softmax(activation) for activation in activations]) - seq_y
            self.backward_pass(deltas, update=True)

            """ LEARNING RATE ADJUSTMENTS """
            if not (iter + 1) % 2000:
                for layer in self.layers:
                    if hasattr(layer, "optimizer") and hasattr(layer.optimizer, "eta"):
                        layer.optimizer.eta /= 2.0

            iter += 1
        return True

    """
    The way to validate this is to try to construct a valid sequence, and display
    it to the user for them to evaluate it.
    """

    def validate(self):
        len_result = 200
        x = utils.int2Onehot(np.random.randint(0, self.len_elem), self.len_elem)
        result = np.zeros((len_result, self.len_elem, 1))
        result[0] = x
        for t in range(len_result-1):
            output_seq = self.forward_pass_batch(result[-20:])
            p = utils.softmax(output_seq[t if t<self.len_seq else -1])
            output = utils.int2Onehot(np.random.choice(range(self.len_elem), p=p.ravel()), self.len_elem)
            result[t+1] = output
        # print([self.index2Word[utils.onehot2Int(label)] for label in result])
        print(" ".join([str(utils.onehot2Int(label)) for label in result]))
