import numpy as np


class Adadelta:
    def __init__(self):
        self.avg_grad_sq = None
        self.avg_delta_sq = None
        # self.avg_w_grad_sq = 0.0
        # self.avg_b_grad_sq = 0.0
        # self.avg_w_delta_sq = 0.01
        # self.avg_b_delta_sq = 0.01
        # Prevents division by zero
        self.eps = 1e-8
        # Determines how much of previous average squared gradients to keep
        self.gamma = 0.9

    def delta(self, *gradients):
        if not self.avg_grad_sq:
            self.avg_grad_sq = [gradient * gradient for gradient in gradients]
            self.avg_delta_sq = [gradient * gradient for gradient in gradients]
        assert len(gradients) == len(self.avg_grad_sq), "Inconsistent gradients list length"
        deltas = []
        for i in range(len(gradients)):
            self.avg_grad_sq[i] = self.gamma * self.avg_grad_sq[i] + (1.0 - self.gamma) * gradients[i] * gradients[i]
            delta = np.sqrt(self.avg_delta_sq[i]) / (np.sqrt(self.avg_grad_sq[i]) + self.eps) * gradients[i]
            deltas.append(delta)
            self.avg_delta_sq[i] = self.gamma * self.avg_delta_sq[i] + (1.0 - self.gamma) * delta * delta
        return deltas