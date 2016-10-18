import numpy as np
import theano
import abc
from theano import tensor as tensor

from ann.Layers import HiddenLayer, LinearRegressionLayer, LogisticRegressionLayer


def _add_hidden_layers(seed, input_layer, network_specification):
    hidden_layers = []
    input_stream = input_layer
    for index, layer in enumerate(network_specification):
        if 0 == index or index == len(network_specification) - 1:
            continue

        hidden_layer = HiddenLayer(seed=seed,
                                   input_stream=input_stream,
                                   n_in=network_specification[index - 1],
                                   n_out=network_specification[index])
        hidden_layers.append(hidden_layer)
        input_stream = hidden_layer.output_stream
    return hidden_layers


def _verify_network_specification(network_specification):
    if len(network_specification) < 3:
        raise InvalidNetworkError


def _verify_dataset(dataset):
    if len(dataset) < 1:
        raise NoDatasetFoundError
    if len(dataset[0]) != 2:
        raise InvalidDataError


def _collection_params(hidden_layers, output_layer):
    params = []
    for layer in hidden_layers:
        params = params + layer.params
    params = params + output_layer.params
    return params


def _prepare_data(dataset):
    try:
        dataset = dataset.T

        # TODO: I need to convert tolist before to array because lowest level is list? The hell numpy
        data_x = np.asarray(dataset[0].tolist())
        data_y = np.asarray(dataset[1].tolist())

        shared_x = theano.shared(data_x, borrow=True)
        shared_y = theano.shared(data_y, borrow=True)

        return shared_x, shared_y, len(data_x)
    except AttributeError:
        raise NoNumpyArrayError


class MultiLayerPerceptron(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, seed, network_specification):
        # Assert inputs
        _verify_network_specification(network_specification)
        self.network_specification = network_specification

        # Build network
        self._input_vector = tensor.matrix('x')
        self._hidden_layers = _add_hidden_layers(seed, self._input_vector, network_specification)
        self._output_layer = self.add_output_layer(seed, self._hidden_layers, network_specification)
        self._output_vector = self.add_output_vector()
        self._params = _collection_params(self._hidden_layers, self._output_layer)

        # Functions
        self._predict = self.prediction_function(self._input_vector, self._output_layer)

    def train(self, training_set, iterations=10, learning_rate=0.1, batch_size=1):
        _verify_dataset(training_set)
        self.verify_dimensions(training_set, self.network_specification)
        self._data_x, self._data_y, self._data_points = _prepare_data(training_set)

        cost_function = self._output_layer.cost(self._output_vector)
        gradients = [tensor.grad(cost_function, param) for param in self._params]
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self._params, gradients)]
        index = tensor.lscalar()
        train_model = theano.function(inputs=[index],
                                      outputs=cost_function,
                                      updates=updates,
                                      givens={
                                          self._input_vector: self._data_x[index * batch_size:(index + 1) * batch_size],
                                          self._output_vector: self._data_y[index * batch_size:(index + 1) * batch_size]
                                      })

        for i in range(iterations):
            print "iteration " + str(i + 1) + "/" + str(iterations)
            for batch in range(self._data_points / batch_size):
                train_model(batch)

    def test(self, test_set, batch_size=1):
        _verify_network_specification(self.network_specification)
        _verify_dataset(test_set)
        self.verify_dimensions(test_set, self.network_specification)

        data_x, data_y, zz = _prepare_data(test_set)

        index = tensor.lscalar()
        test_model = theano.function(inputs=[index],
                                     outputs=self._output_layer.error(self._output_vector),
                                     givens={
                                         self._input_vector: data_x[index * batch_size:(index + 1) * batch_size],
                                         self._output_vector: data_y[index * batch_size:(index + 1) * batch_size]
                                     })

        test_loss = [test_model(i) for i in range(len(test_set))]
        return np.mean(test_loss) * 100

    def predict(self, input_vector):
        return self._predict(input_vector)

    @abc.abstractmethod
    def verify_dimensions(self, dataset, network_specification):
        pass

    @abc.abstractmethod
    def add_output_layer(self, seed, hidden_layers, network_specification):
        pass

    @abc.abstractmethod
    def add_output_vector(self):
        pass

    @abc.abstractmethod
    def prediction_function(self, input_layer, output_layer):
        pass


class MultiLayerPerceptronRegressor(MultiLayerPerceptron):
    def __init__(self, seed, network_specification):
        MultiLayerPerceptron.__init__(self, seed, network_specification)

    def verify_dimensions(self, dataset, network_specification):
        input_size = len(dataset[0][0])
        if input_size != network_specification[0]:
            raise InvalidDimensionError

        output_size = len(dataset[0][1])
        if output_size != network_specification[-1]:
            raise InvalidDimensionError

    def add_output_layer(self, seed, hidden_layers, network_specification):
        return LinearRegressionLayer(seed=seed,
                                     input_stream=hidden_layers[-1].output_stream,
                                     n_in=network_specification[-2],
                                     n_out=network_specification[-1])

    def add_output_vector(self):
        return tensor.matrix('y')

    def prediction_function(self, input_layer, output_layer):
        return theano.function(inputs=[input_layer],
                               outputs=output_layer.predict()[0]
                               )


class MultiLayerPerceptronClassifier(MultiLayerPerceptron):
    def __init__(self, seed, network_specification):
        MultiLayerPerceptron.__init__(self, seed, network_specification)

    def verify_dimensions(self, dataset, network_specification):
        input_size = len(dataset[0][0])
        if input_size != network_specification[0]:
            raise InvalidDimensionError

        number_of_classes = np.unique(dataset.T[1].tolist()).size
        if number_of_classes != network_specification[-1]:
            raise InvalidDimensionError

    def add_output_layer(self, seed, hidden_layers, network_specification):
        return LogisticRegressionLayer(seed=seed,
                                       input_stream=hidden_layers[-1].output_stream,
                                       n_in=network_specification[-2],
                                       n_out=network_specification[-1])

    def add_output_vector(self):
        return tensor.lvector('y')

    def prediction_function(self, input_layer, output_layer):
        return theano.function(inputs=[input_layer],
                               outputs=output_layer.predict()
                               )


class InvalidNetworkError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Network must consist of at least an input layer, 1 hidden layer, and an output layer')


class InvalidDataError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Data set should be formatted as follows: [[[input],[output]],[[input],[output]]]')


class NoNumpyArrayError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Data should consist of numpy arrays')


class NoDatasetFoundError(Exception):
    def __init__(self):
        Exception.__init__(self, 'No input data has been found')


class InvalidDimensionError(Exception):
    def __init__(self):
        Exception.__init__(self, 'The input and output sizes of the data set do not match with the specified network')
