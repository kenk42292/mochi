import numpy as np


class RMSProp:
    def __init__(self, eta):
        self.eta = eta
        self.avg_grad_sq = None
        # Prevents division by zero
        self.eps = 1e-8
        # Determines how much of previous average squared gradients to keep
        self.gamma = 0.9

    def delta(self, *gradients):
        if not self.avg_grad_sq:
            self.avg_grad_sq = [gradient * gradient for gradient in gradients]
        else:
            assert len(gradients) == len(self.avg_grad_sq), "Inconsistent gradients list length"
            for i in range(len(gradients)):
                self.avg_grad_sq[i] = self.gamma * self.avg_grad_sq[i] + (1.0 - self.gamma) * gradients[i] * gradients[
                    i]
        return [(self.eta / (np.sqrt(self.avg_grad_sq[i]) + self.eps)) * gradients[i] for i in range(len(gradients))]
