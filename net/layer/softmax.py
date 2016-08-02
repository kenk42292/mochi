from neurallayer import  NeuralLayer


class SoftMax(NeuralLayer):
    def __init__(self, input_dim, output_dim):
        NeuralLayer.__init__(self, input_dim, output_dim)
        print("instantiating SoftMax")

    def feed_forward(self, input):
        return

    def back_prop(self, delta):
        return delta