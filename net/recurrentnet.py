from neuralnet import NeuralNet
from data_iter import SeqIter


class RecurrentNet(NeuralNet):

    """
    A Neural Network that supports Recurrent layers.
    The only difference between a feedforward net and a recurrent net is that, in a recurrent net,
    the mini-batches are adjacent, and in order. In other words, they are SEQUENCES.
    Of course, using the feedforward mini-batches as sequences takes away the ability of
    a recurrent net to do mini-batch gradient descent with batches of sequences.
    But that's alright - in fact, that's usually how it's done.
    """
    def __init__(self, layers, layers_file):
        print("Creating Recurrent Net")
        NeuralNet.__init__(self, layers, layers_file)
        self.data_iter_type = SeqIter