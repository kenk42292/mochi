import numpy as np
import math


class Adam:
    def __init__(self, eta, beta1=0.9, beta2=0.999):
        self.eta = eta
        self.mean_grads = None
        self.var_grads = None
        # Prevents division by zero
        self.eps = 1e-8
        # Decay rates for mean and variance of gradients
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def delta(self, *gradients):
        self.t += 1
        if not self.mean_grads:
            self.mean_grads = [0] * len(gradients)
            self.var_grads = [0] * len(gradients)
        for i in range(len(gradients)):
            self.mean_grads[i] = self.beta1 * self.mean_grads[i] + (1.0 - self.beta1) * gradients[i]
            self.var_grads[i] = self.beta2 * self.var_grads[i] + (1.0 - self.beta2) * gradients[i] * gradients[i]
        mean_corrected = [mean_grad / (1.0 - self.beta1 ** self.t) for mean_grad in self.mean_grads]
        var_corrected = [var_grad / (1.0 - self.beta2 ** self.t) for var_grad in self.var_grads]
        deltas = [self.eta * m / (np.sqrt(v) + self.eps) for m, v in zip(mean_corrected, var_corrected)]
        return deltas
