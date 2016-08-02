from neuralnet import NeuralNet
from data_iter import RandBatchIter


class FeedForwardNet(NeuralNet):

    def __init__(self, layers, layers_file):
        print("Creating FeedForward Net")
        NeuralNet.__init__(self, layers, layers_file)
        self.data_iter_type = RandBatchIter
