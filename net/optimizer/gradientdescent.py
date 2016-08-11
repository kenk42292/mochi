
class GradientDescent:

    def __init__(self, eta):
        self.eta = eta

    def delta(self, *gradients):
        return [self.eta*gradient for gradient in gradients]
