
import numpy as np

class Adadelta:
    def __init__(self):
        self.avg_w_grad_sq = 0.0
        self.avg_b_grad_sq = 0.0
        self.avg_w_delta_sq = 0.01
        self.avg_b_delta_sq = 0.01
        # Prevents division by zero
        self.eps = 1e-8
        # Determines how much of previous average squared gradients to keep
        self.gamma = 0.9

    def delta(self, weight_gradient, bias_gradient):
        self.avg_w_grad_sq = self.gamma*self.avg_w_grad_sq + (1.0-self.gamma)*weight_gradient*weight_gradient
        self.avg_b_grad_sq = self.gamma*self.avg_b_grad_sq + (1.0-self.gamma)*bias_gradient*bias_gradient

        delta_w = ((np.sqrt(self.avg_w_delta_sq)+self.eps)/(np.sqrt(self.avg_w_grad_sq)+self.eps)) * weight_gradient
        delta_b = ((np.sqrt(self.avg_b_delta_sq)+self.eps)/(np.sqrt(self.avg_b_grad_sq) + self.eps)) * bias_gradient

        self.avg_w_delta_sq = self.gamma*self.avg_w_delta_sq + (1.0-self.gamma)*delta_w*delta_w
        self.avg_b_delta_sq = self.gamma*self.avg_b_delta_sq + (1.0-self.gamma)*delta_b*delta_b

        return delta_w, delta_b


