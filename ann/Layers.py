import numpy as np
import theano
from theano import tensor as tensor
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from operator import mul


def _init_bias(n_out):
    b_values = np.zeros((n_out,), dtype=theano.config.floatX)
    return theano.shared(value=b_values, name='b', borrow=True)


def _init_weights(n_in, n_out, shape, seed):
    rng = np.random.RandomState(seed)
    w = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=shape
        ),
        dtype=theano.config.floatX
    )
    return theano.shared(value=w, name='w', borrow=True)


def _init_params(n_in, n_out, shape, seed, n_out_shape):
    weights = _init_weights(n_in, n_out, shape, seed)
    bias = _init_bias(n_out_shape)
    return weights, bias


class InputLayer(object):
    def __init__(self, input_size):
        self.size = reduce(mul, input_size)
        self.output_shape = input_size
        self.output_stream = tensor.matrix('x')
        self.feature_maps = 1

    def reshape(self, batch_size):
        return self.output_stream.reshape((batch_size, 1, self.output_shape[0], self.output_shape[1]))

    def verify_dimensions(self, data_set):
        if len(data_set[0][0]) != self.size:
            raise InvalidDimensionError


class HiddenLayer(object):
    def __init__(self, size):
        self.size = size

    def connect(self, seed, input_layer):
        n_in = input_layer.size
        weights, bias = _init_params(n_in, self.size, (n_in, self.size), seed, self.size)

        self.output_stream = tensor.tanh(tensor.dot(input_layer.output_stream.flatten(2), weights) + bias)
        self.params = [weights, bias]


class LinearRegressionLayer(object):
    def __init__(self, size):
        self.size = size
        self.output_vector = tensor.matrix('y')

    def connect(self, seed, input_layer):
        n_in = input_layer.size
        weights, bias = _init_params(n_in, self.size, (n_in, self.size), seed, self.size)

        self.y_prediction = tensor.dot(input_layer.output_stream, weights) + bias
        self.params = [weights, bias]

    def predict(self):
        return self.y_prediction

    def error(self, y):
        return tensor.mean((self.y_prediction - y) ** 2)

    def cost(self, y):
        return tensor.mean((self.y_prediction - y) ** 2)  # TODO: add l1 and l2 reg

    def verify_dimensions(self, data_set):
        output_size = len(data_set[0][1])
        if output_size != self.size:
            raise InvalidDimensionError


class LogisticRegressionLayer(object):
    def __init__(self, size):
        self.size = size
        self.output_vector = tensor.lvector('y')

    def connect(self, seed, input_layer):
        n_in = input_layer.size
        weights, bias = _init_params(n_in, self.size, (n_in, self.size), seed, self.size)

        self.p_y_given_x = tensor.nnet.softmax(tensor.dot(input_layer.output_stream, weights) + bias)
        self.y_prediction = tensor.argmax(self.p_y_given_x, axis=1)
        self.params = [weights, bias]

    def predict(self):
        return self.y_prediction

    def error(self, y):
        return tensor.mean(tensor.neq(self.y_prediction, y))

    def cost(self, y):
        return -tensor.mean(tensor.log(self.p_y_given_x)[tensor.arange(y.shape[0]), y])  # TODO: add l1 and l2 reg

    def verify_dimensions(self, data_set):
        number_of_classes = np.unique(data_set.T[1].tolist()).size
        if number_of_classes != self.size:
            raise InvalidDimensionError


def _get_input_stream(batch_size, input_layer):
    return input_layer.reshape(batch_size) if isinstance(input_layer, InputLayer) else input_layer.output_stream


def _get_in_out_size(n_in_feature_maps, feature_map, filter_shape, pool_size):
    return n_in_feature_maps * np.prod(filter_shape), feature_map * np.prod(filter_shape) // np.prod(pool_size)


def _get_output_shape(input_shape, filter_shape, pool_size):
    return (input_shape[0] - filter_shape[0] + 1) / pool_size[0], (input_shape[1] - filter_shape[1] + 1) / pool_size[1]


class LeNetConvPoolLayer(object):
    def __init__(self, feature_map, filter, pool):
        self.feature_maps = feature_map
        self.filter_shape = filter
        self.pool_size = pool

    # TODO: remove batch_size and refactor
    def connect(self, seed, input_layer, batch_size=500):
        n_in_feature_maps = input_layer.feature_maps
        input_shape = input_layer.output_shape

        all_filters_shape = (self.feature_maps, n_in_feature_maps, self.filter_shape[0], self.filter_shape[1])
        n_in, n_out = _get_in_out_size(n_in_feature_maps, self.feature_maps, self.filter_shape, self.pool_size)
        weights, bias = _init_params(n_in, n_out, all_filters_shape, seed, self.feature_maps)
        self.params = [weights, bias]

        self.output_stream = self._set_output_stream(batch_size, input_layer)
        self.output_shape = _get_output_shape(input_shape, self.filter_shape, self.pool_size)
        self.size = self.feature_maps * self.output_shape[0] * self.output_shape[1]

    def _set_output_stream(self, batch_size, input_layer):
        # convolve input feature maps with filters
        convolve_out = conv2d(
            input=_get_input_stream(batch_size, input_layer),
            filters=self.params[0]
        )
        # pool each feature map individually, using max pooling
        pooled_out = pool.pool_2d(
            input=convolve_out,
            ds=self.pool_size,
            ignore_border=True
        )
        # reshape bias to (1, n_filters, 1, 1) to broadcast to all batches and feature maps
        return tensor.tanh(pooled_out + self.params[1].dimshuffle('x', 0, 'x', 'x'))


class InvalidDimensionError(Exception):
    def __init__(self):
        Exception.__init__(self, 'The input and output sizes of the data set do not match with the specified network')
