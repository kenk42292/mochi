import numpy as np


class Momentum:
    def __init__(self, eta, gamma=0.9):
        self.eta = eta
        self.mean_gradients = None
        # Prevents division by zero
        self.eps = 1e-8
        # Decay rates for mean of gradients
        self.gamma = gamma

    def delta(self, *gradients):
        if self.mean_gradients is None:
            self.mean_gradients = list(gradients)
        else:
            for i in range(len(gradients)):
                self.mean_gradients[i] = self.gamma * self.mean_gradients[i] + (1.0 - self.gamma) * gradients[i]

        return [self.eta*grad for grad in self.mean_gradients]
