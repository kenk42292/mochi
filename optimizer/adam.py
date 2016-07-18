
import numpy as np
import math

class Adam:
    def __init__(self, eta=1e-8, beta1=0.9, beta2=0.999):
        self.eta = eta
        self.mean_w_grad = 0.0
        self.mean_b_grad = 0.0
        self.var_w_grad = 0.0
        self.var_b_grad = 0.0
        #Prevents division by zero
        self.eps = 1e-8
        #Decay rates for mean and variance of gradients
        self.beta1 = beta1
        self.beta2 = beta2

    def delta(self, weight_gradient, bias_gradient):
        self.mean_w_grad = self.beta1 * self.mean_w_grad + (1.0 - self.beta1) * weight_gradient
        self.mean_b_grad = self.beta1 * self.mean_b_grad + (1.0 - self.beta1) * bias_gradient
        self.var_w_grad = self.beta2 * self.var_w_grad + (1.0 - self.beta2) * weight_gradient * weight_gradient
        self.var_b_grad = self.beta2 * self.var_b_grad + (1.0 - self.beta2) * bias_gradient * bias_gradient
        mean_w_corrected = self.mean_w_grad/(1.0-self.beta1)
        mean_b_corrected = self.mean_b_grad/(1.0-self.beta1)
        var_w_corrected = self.var_w_grad/(1.0-self.beta2)
        var_b_corrected = self.var_b_grad/(1.0-self.beta2)
        delta_w = self.eta*mean_w_corrected/(np.sqrt(var_w_corrected)+self.eps)
        delta_b = self.eta*mean_b_corrected/(np.sqrt(var_b_corrected)+self.eps)
        return delta_w, delta_b

