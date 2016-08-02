import numpy as np
import utils
import time
import cPickle as pickle

from data_iter import RandBatchIter

"""
A feed-forward-only neural net. Within LAYERS, can take only feed-forward layers (No recurrent layers)
However, unlike a recurrent net, supports mini-batch training.
"""


class NeuralNet:
    def __init__(self, layers, layers_file):
        self.layers = layers
        self.num_layers = len(layers)
        self.layers_file = layers_file
        self.data_iter_type = RandBatchIter

    def train(self, training_data, validation_data, niter=100, batch_size=100):
        iter = 0
        t = time.time()
        data_iter = self.data_iter_type(training_data, batch_size)
        while iter < niter:

            """ BATCH ASSIGNMENT """
            batch_x, batch_y = data_iter.next();

            """ TRAINING EVALUATION """
            if not (iter + 1) % 1:
                print("######################################################################################")
                print("iter: %d" % iter)
                print("time elased: " + str(time.time() - t))
                print("training loss: " + str(self.loss(batch_x, batch_y)))
                print("training validation rate %0.17f" % self.validate(batch_x, batch_y))
                with open(self.layers_file, "wb") as handle:
                    pickle.dump(self.layers, handle)
                t = time.time()

            """ TRAINING """
            print('---------------------------------')
            activations = self.forward_pass(batch_x)
            deltas = np.array([utils.softmax(activation) for activation in activations]) - batch_y
            self.backward_pass(deltas, update=True)

            """ LEARNING RATE ADJUSTMENTS """
            if not (iter + 1) % 2000:
                for layer in self.layers:
                    if hasattr(layer, "optimizer") and hasattr(layer.optimizer, "eta"):
                        layer.optimizer.eta /= 2.0

            iter += 1
        val_batch_x = np.array([sample[0] for sample in validation_data])
        val_batch_y = np.array([sample[1] for sample in validation_data])
        print("validation rate: %0.17f" % (self.validate(val_batch_x, val_batch_y, 100)))
        return True

    def forward_pass(self, activations):
        for i in range(self.num_layers):
            activations = self.layers[i].feed_forward(activations)
        return activations

    def backward_pass(self, deltas, update=True):
        for i in range(self.num_layers)[::-1]:
            deltas = self.layers[i].back_prop(deltas, update)

    def predict(self, batch_x):
        activations = self.forward_pass(batch_x)
        return np.array([max(range(len(activation)), key=lambda i: activation[i]) for activation in activations])

    def validate(self, batch_x, batch_y, forced_batch_size=False):
        if forced_batch_size:
            #print("FORCED BATCH SIZE!!")
            total = 0.0
            num_batches = len(batch_x) // forced_batch_size
            #print(num_batches)
            for i in range(num_batches):
                partial_batch_rate = self.validate(batch_x[i * forced_batch_size: (i + 1) * forced_batch_size],
                                                   batch_y[i * forced_batch_size:(i + 1) * forced_batch_size])
                #print("partial batch rate: " + str(partial_batch_rate))
                total += partial_batch_rate
            return float(total) / num_batches
        predictions = self.predict(batch_x)
        num_correct = 0
        for prediction, y in zip(predictions, batch_y):
            if prediction == (y if (type(y) is np.int64) else utils.onehot2Int(y)):
                num_correct += 1
        return float(num_correct) / len(batch_x)

    def loss(self, batch_x, batch_y):
        outputs = np.array([utils.softmax(activation) for activation in self.forward_pass(batch_x)])
        return -float(np.sum(batch_y * np.log(outputs)))

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
        activation = self.forward_pass([x])
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
                estimated_param_grad = ((grad_plus - grad_minus) / (2 * d))
                model_param_grad = model_grad[index_iter.multi_index]
                relative_error = np.abs(model_param_grad - estimated_param_grad) \
                                 / (np.abs(model_param_grad) + np.abs(estimated_param_grad))
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
