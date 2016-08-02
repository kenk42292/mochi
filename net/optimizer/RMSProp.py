
import numpy as np

class RMSProp:
    def __init__(self, eta):
        self.eta = eta
        self.avg_w_grad_sq = None
        self.avg_b_grad_sq = None
        #Prevents division by zero
        self.eps = 1e-8
        #Determines how much of previous average squared gradients to keep
        self.gamma = 0.9

    def delta(self, weight_gradient, bias_gradient):
        if self.avg_w_grad_sq is None:
            self.avg_w_grad_sq = weight_gradient * weight_gradient
            self.avg_b_grad_sq = bias_gradient * bias_gradient
        else:
            self.avg_w_grad_sq = self.gamma*self.avg_w_grad_sq + (1.0-self.gamma)*weight_gradient*weight_gradient
            self.avg_b_grad_sq = self.gamma*self.avg_b_grad_sq + (1.0-self.gamma)*bias_gradient*bias_gradient

        delta_w = (self.eta/(np.sqrt(self.avg_w_grad_sq)+self.eps)) * weight_gradient
        delta_b = (self.eta/(np.sqrt(self.avg_b_grad_sq) + self.eps)) * bias_gradient

        print("RMSProp update largest magnitudes are: (%f, %f) with sums of (%f, %f) and abs sums of (%f, %f)"
              % (max([np.max(delta_w), np.min(delta_w)], key=abs),
                 max([np.max(delta_b), np.min(delta_b)], key=abs),
                 np.sum(delta_w),
                  np.sum(delta_b),
                  np.sum(np.abs(delta_w)),
                  np.sum(np.abs(delta_b))))

        return delta_w, delta_b
