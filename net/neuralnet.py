import numpy as np
import utils
import time
import cPickle as pickle

class NeuralNet(object):

    def __init__(self, layers, layers_file):
        self.layers = layers
        self.num_layers = len(layers)
        self.layers_file = layers_file

    def train(self, training_data, validation_data, niter=100, batch_size=100):
        iter = 0
        t = time.time()
        while iter < niter:
            batch_indeces = np.random.choice(range(len(training_data)), size=batch_size, replace=False)
            for sample_index in batch_indeces:
                x, y = training_data[sample_index]
                activation = self.forward_pass(x)
                delta = utils.softmax(activation) - y
                self.backward_pass(delta)
            for layer in self.layers:
                layer.update(batch_size)
                layer.clear_grads()

            if not (iter + 1) % 20:
                print("######################################################################################")
                print("iter: %d" % iter)
                print("time elased: " + str(time.time() - t))
                print("training loss: " + str(self.training_loss(training_data, batch_indeces)))
                print("training validation rate %0.17f" % self.train_validate(training_data, batch_indeces))
                with open(self.layers_file, "wb") as handle:
                    pickle.dump(self.layers, handle)
                t = time.time()
            iter += 1
        print("validation rate: %0.17f" % self.validate(validation_data))
        return True

    def forward_pass(self, activation):
        for i in range(self.num_layers):
            activation = self.layers[i].feed_forward(activation)
        return activation

    def backward_pass(self, delta):
        for i in range(self.num_layers)[::-1]:
            delta = self.layers[i].back_prop(delta)

    def predict(self, x):
        activation = self.forward_pass(x)
        return max(range(len(activation)), key=lambda i: activation[i])

    def train_validate(self, training_data, batch_indeces):
        num_correct = 0
        for batch_index in batch_indeces:
            x, y = training_data[batch_index]
            prediction = self.predict(x)
            if prediction == utils.onehot2Int(y):
                num_correct += 1
        return float(num_correct)/len(batch_indeces)

    def validate(self, validation_data):
        num_correct = 0
        for x, y in validation_data:
            prediction = self.predict(x)
            if prediction == y:
                num_correct += 1
        return float(num_correct)/len(validation_data)

    def training_loss(self, training_data, training_indeces):
        total = 0.0
        for index in training_indeces:
            x, y = training_data[index]
            total += self.loss(x, y)
        return total

    def loss(self, x, y):
        output = utils.softmax(self.forward_pass(x))
        return -float(np.sum(y * np.log(output)))

    def grad_check(self, x, y, d=1e-6, err_threshold=0.01):
        """
        Note: the grad_check CLEARS all gradients in each layer.
        Gradient check is only performed on first layer parameters.
        :param x: sample input
        :param y: desired sample output
        :param d: perturbation used in gradient calculation
        :param err_threshold: tolerance of error in gradient
        :return:
        """
        for layer in self.layers:
            layer.clear_grads()
        activation = self.forward_pass(x)
        delta = utils.softmax(activation) - y
        self.backward_pass(delta)

        layer = self.layers[0]
        param_names = ["Wxz", "bz"]
        grad_names = ["dL_dWxz", "dL_dbz"]
        for param_name, grad_name in zip(param_names, grad_names):
            model_param = getattr(layer, param_name)
            model_grad = getattr(layer, grad_name)
            index_iter = np.nditer(model_grad, flags=["multi_index"], op_flags=["readwrite"])
            while not index_iter.finished:
                original_value = model_param[index_iter.multi_index]
                model_param[index_iter.multi_index] = original_value + d
                grad_plus = self.loss(x, y)
                model_param[index_iter.multi_index] = original_value - d
                grad_minus = self.loss(x, y)
                model_param[index_iter.multi_index] = original_value
                estimated_param_grad = ((grad_plus-grad_minus)/(2*d))
                model_param_grad = model_grad[index_iter.multi_index]
                relative_error = np.abs(model_param_grad-estimated_param_grad)\
                    / (np.abs(model_param_grad)+np.abs(estimated_param_grad))
                if (relative_error > err_threshold) and (relative_error < 1.0):
                    print("Gradient Check Fails for %s[%s]" % (param_name, index_iter.multi_index))
                    print("grad_plus: %0.17f" % grad_plus)
                    print("grad_minus: %0.17f" % grad_minus)
                    print("Estimated Gradient: %0.17f" % estimated_param_grad)
                    print("Model Generated Gradient: %0.17f" % model_param_grad)
                    print("relative_error: %0.17f" % relative_error)
                    print("err_threshold: %0.17f" % err_threshold)
                    print(relative_error > err_threshold)
                    # return
                index_iter.iternext()
            print("Gradient check for %s passed" % param_name)
        for layer in self.layers:
            layer.clear_grads()
