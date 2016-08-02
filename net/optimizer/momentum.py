import numpy as np

class Momentum:
    def __init__(self, eta, gamma=0.9):
        self.eta = eta
        self.mean_w_grad = None
        self.mean_b_grad = None
        #Prevents division by zero
        self.eps = 1e-8
        #Decay rates for mean of gradients
        self.gamma = gamma

    def delta(self, weight_gradient, bias_gradient):
        if self.mean_w_grad is None and self.mean_b_grad is None:
            self.mean_w_grad, self.mean_b_grad = weight_gradient, bias_gradient
        else:
            self.mean_w_grad = self.gamma * self.mean_w_grad + (1.0-self.gamma) * weight_gradient
            self.mean_b_grad = self.gamma * self.mean_b_grad + (1.0-self.gamma) * bias_gradient
        # print(np.sum(np.abs(self.mean_w_grad)))
        return self.eta*self.mean_w_grad, self.eta*self.mean_b_grad

