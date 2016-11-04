import numpy as np


class Adagrad:
    def __init__(self, eta):
        self.eta = eta
        self.sum_grad_sq = None
        self.eps = 1e-8

    def delta(self, *gradients):
        if not self.sum_grad_sq:
            self.sum_grad_sq = [gradient * gradient for gradient in gradients]
            return [self.eta * gradient for gradient in gradients]
        deltas = []
        for i in range(len(gradients)):
            self.sum_grad_sq[i] += gradients[i] * gradients[i]
            deltas.append(self.eta / (np.sqrt(self.sum_grad_sq[i]) + self.eps) * gradients[i])
        return deltas
