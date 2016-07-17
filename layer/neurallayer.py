
class NeuralLayer(object):

    def __init__(self, input_dim, output_dim, categorical_input=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.categorical_input = categorical_input

    def feed_forward(self, input):
        print("not implemented")

    def back_prop(self, delta):
        print("not implemened")

