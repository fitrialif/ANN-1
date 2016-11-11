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
    return [weights, bias]


class InputLayer(object):
    def __init__(self, input_size):
        """
        Construct an input layer that contains the input data format

        :param input_size: List that specifies the dimensions of the input data
        """
        self.size = reduce(mul, input_size)
        self.output_shape = input_size
        self.output_stream = tensor.matrix('x')
        self.feature_maps = 1

    def reshape(self, batch_size):
        """
        For convolution layers, reshape the input dimensions based on the batch_size

        :param batch_size: Batch size used for either training or testing the network
        :return: Reshaped output data stream
        """
        return self.output_stream.reshape((batch_size, 1, self.output_shape[0], self.output_shape[1]))

    def verify_dimensions(self, data_set):
        """
        Verify if provided data fits in the network, throw InvalidDimensionError exception if not

        :param data_set: Data that is going through the network
        """
        if len(data_set[0][0]) != self.size:
            raise InvalidDimensionError


class HiddenLayer(object):
    def __init__(self, size):
        """
        Construct a bare bone hidden layer of a given size

        :param size: Number of neurons in this layer
        """
        self.size = size
        self.params = None
        self.output_stream = None
        self.input_layer = None

    def init_weights(self, seed, input_layer):
        """
        Initialize weights of hidden layer

        :param seed: A seed used to initialize the weights and biases of the layer randomly
        :param input_layer: Previous layer that feeds into this one. Required to determine the amount of weights and
        biases that have to be initialized
        :return: Return all initialized parameters
        """
        n_in = input_layer.size
        self.input_layer = input_layer
        self.params = _init_params(n_in, self.size, (n_in, self.size), seed, self.size)
        return self.params

    def connect(self):
        """
        Define the output data stream as a function of the input data stream, the weights and biases, and pushing it
        through a tanh function. The input is flattened so it can connect with a convolution layer
        """
        weights = self.params[0]
        bias = self.params[1]
        self.output_stream = tensor.tanh(tensor.dot(self.input_layer.output_stream.flatten(2), weights) + bias)


class LinearRegressionLayer(object):
    def __init__(self, size):
        """
        Construct a bare bone linear regression layer of a given size

        :param size: Number of output values
        """
        self.size = size
        self.output_vector = tensor.matrix('y')
        self.params = None
        self.y_prediction = None
        self.input_layer = None

    def init_weights(self, seed, input_layer):
        """
        Initialize weights of linear regression layer

        :param seed: A seed used to initialize the weights and biases of the layer randomly
        :param input_layer: Previous layer that feeds into this one. Required to determine the amount of weights and
        biases that have to be initialized
        :return: Return all initialized parameters
        """
        n_in = input_layer.size
        self.input_layer = input_layer
        self.params = _init_params(n_in, self.size, (n_in, self.size), seed, self.size)
        return self.params

    def connect(self):
        """
        Define the output data stream as a prediction function
        """
        weights = self.params[0]
        bias = self.params[1]
        self.y_prediction = tensor.dot(self.input_layer.output_stream, weights) + bias

    def predict(self):
        """
        :return: Symbolic variable that represents the predicted value
        """
        return self.y_prediction

    def error(self, y):
        """
        Calculate and return the squared error for some given data

        :param y: Known verification data
        :return: Squared error of predicted and known data
        """
        return tensor.mean((self.y_prediction - y) ** 2)

    def cost(self, y):
        """
        Determine the cost for some data when compared to its known values

        :param y: Known verification data
        :return: Squared error of predicted and known data
        """
        return tensor.mean((self.y_prediction - y) ** 2)  # TODO: add l1 and l2 reg

    def verify_dimensions(self, data_set):
        """
        Verify if provided data fits in the network, throw InvalidDimensionError exception if not

        :param data_set: Data that is going through the network
        """
        output_size = len(data_set[0][1])
        if output_size != self.size:
            raise InvalidDimensionError


class LogisticRegressionLayer(object):
    def __init__(self, size):
        """
        Construct a bare bone logistic regression layer of a given size

        :param size: Number of classes
        """
        self.size = size
        self.output_vector = tensor.lvector('y')
        self.params = None
        self.p_y_given_x = None
        self.y_prediction = None
        self.input_layer = None

    def init_weights(self, seed, input_layer):
        """
        Initialize weights of linear regression layer

        :param seed: A seed used to initialize the weights and biases of the layer randomly
        :param input_layer: Previous layer that feeds into this one. Required to determine the amount of weights and
        biases that have to be initialized
        :return: Return all initialized parameters
        """
        n_in = input_layer.size
        self.params = _init_params(n_in, self.size, (n_in, self.size), seed, self.size)
        self.input_layer = input_layer
        return self.params

    def connect(self):
        """
        Define the output data stream as a prediction function
        """
        weights = self.params[0]
        bias = self.params[1]
        self.p_y_given_x = tensor.nnet.softmax(tensor.dot(self.input_layer.output_stream, weights) + bias)
        self.y_prediction = tensor.argmax(self.p_y_given_x, axis=1)

    def predict(self):
        """
        :return: Symbolic variable that represents the predicted value
        """
        return self.y_prediction

    def error(self, y):
        """
        Calculate and return the error for some given data

        :param y: Known verification data
        :return: Average error rate of wrongly classified predicted compared to its known cass
        """
        return tensor.mean(tensor.neq(self.y_prediction, y))

    def cost(self, y):
        """
        Determine the cost for some data when compared to its known values

        :param y: Known verification data
        :return: Average error rate of wrongly classified predicted compared to its known cass
        """
        return -tensor.mean(tensor.log(self.p_y_given_x)[tensor.arange(y.shape[0]), y])  # TODO: add l1 and l2 reg

    def verify_dimensions(self, data_set):
        """
        Verify if provided data fits in the network, throw InvalidDimensionError exception if not

        :param data_set: Data that is going through the network
        """
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
    def __init__(self, feature_map, filter_shape, pool_size):
        """
        Construct a bare bone convoltuion layer

        :param feature_map: Number of feature maps in that are used in this layer
        :param filter_shape: Shape of the filters used to calculate the feature maps
        :param pool_size: Size of the pool used to remove noise from the feature maps
        """
        self.feature_maps = feature_map
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.size = None
        self.params = None
        self.output_stream = None
        self.batch_size = None
        self.input_layer = None
        self.output_shape = None

    def init_weights(self, seed, input_layer):
        self.input_layer = input_layer

        self.output_shape = _get_output_shape(input_layer.output_shape, self.filter_shape, self.pool_size)
        self.size = self.feature_maps * self.output_shape[0] * self.output_shape[1]

        n_in_feature_maps = input_layer.feature_maps
        all_filters_shape = (self.feature_maps, n_in_feature_maps, self.filter_shape[0], self.filter_shape[1])
        n_in, n_out = _get_in_out_size(n_in_feature_maps, self.feature_maps, self.filter_shape, self.pool_size)
        weights, bias = _init_params(n_in, n_out, all_filters_shape, seed, self.feature_maps)
        self.params = [weights, bias]
        return self.params

    def connect(self):
        """
        Define the output data stream as a function of the input data stream, the weights and biases, and pushing it
        through a tanh function after it has been convoluted and pooled
        """
        convolve_out = conv2d(input=_get_input_stream(self.batch_size, self.input_layer), filters=self.params[0])
        pooled_out = pool.pool_2d(input=convolve_out, ds=self.pool_size, ignore_border=True)
        self.output_stream = tensor.tanh(pooled_out + self.params[1].dimshuffle('x', 0, 'x', 'x'))
        return self.output_stream

    def set_batch_size(self, batch_size):
        """
        For convolution layers, reshape the input dimensions based on the batch_size

        :param batch_size: Batch size used for either training or testing the network
        """
        self.batch_size = batch_size


class InvalidDimensionError(Exception):
    def __init__(self):
        Exception.__init__(self, 'The input and output sizes of the data set do not match with the specified network')
