import numpy as np


class Adagrad:
    def __init__(self, eta):
        self.eta = eta
        self.sum_w_grad_sq = 0
        self.sum_b_grad_sq = 0
        self.eps = 1e-8

    def delta(self, weight_gradient, bias_gradient):
        self.sum_w_grad_sq += weight_gradient * weight_gradient
        self.sum_b_grad_sq += bias_gradient * bias_gradient
        delta_w = (self.eta/(np.sqrt(self.sum_w_grad_sq)+self.eps)) * weight_gradient
        delta_b = (self.eta / (np.sqrt(self.sum_b_grad_sq) + self.eps)) * bias_gradient
        return delta_w, delta_b
