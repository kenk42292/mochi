
class GradientDescent:

    def __init__(self, eta):
        self.eta = eta

    def delta(self, weight_gradient, bias_gradient):
        return self.eta*weight_gradient, self.eta*bias_gradient
