import numpy as np
import theano
from theano import tensor as tensor


def _init_bias(n_out):
    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
    return theano.shared(value=b_values, name='b', borrow=True)


def _init_weights(n_in, n_out, seed):
    rng = np.random.RandomState(seed)
    w = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )
    return theano.shared(value=w, name='w', borrow=True)


class HiddenLayer(object):
    def __init__(self, seed, input_stream, n_in, n_out, activation=tensor.tanh):
        self.n_in = n_in
        self.n_out = n_out

        self.weights = _init_weights(n_in, n_out, seed)
        self.bias = _init_bias(n_out)
        self.params = [self.weights, self.bias]

        self.output_stream = activation(tensor.dot(input_stream, self.weights) + self.bias)


class LinearRegressionLayer(object):
    def __init__(self, seed, input_stream, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        self.W = _init_weights(n_in, n_out, seed)
        self.b = _init_bias(n_out)
        self.params = [self.W, self.b]

        self.input_stream = input_stream
        self.y_prediction = tensor.dot(input_stream, self.W) + self.b

    def predict(self):
        return self.y_prediction

    def error(self, y):
        return tensor.mean((self.y_prediction - y) ** 2)

    def cost(self, y):
        return tensor.mean((self.y_prediction - y) ** 2)  # TODO: add l1 and l2 reg