import numpy as np
import theano
from theano import tensor as tensor

from ann.Layers import HiddenLayer, LinearRegressionLayer


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


def _add_linear_regression_layer(seed, hidden_layers, network_specification):
    output_layer = LinearRegressionLayer(seed=seed,
                                         input_stream=hidden_layers[-1].output_stream,
                                         n_in=network_specification[-2],
                                         n_out=network_specification[-1])
    return output_layer


def _verify_network_specification(network_specification):
    if len(network_specification) < 3:
        raise InvalidNetworkError


def _verify_dimensions(dataset, network_specification):
    if len(dataset) < 1:
        raise NoDatasetFoundError
    if len(dataset[0]) != 2:
        raise InvalidDataError

    input_size = len(dataset[0][0])
    if input_size != network_specification[0]:
        raise InvalidDimensionError

    output_size = len(dataset[0][1])
    if output_size != network_specification[-1]:
        raise InvalidDimensionError
    pass


def _verify_dimensions2(dataset, network_specification):
    if len(dataset) < 1:
        raise NoDatasetFoundError
    if len(dataset[0]) != 2:
        raise InvalidDataError

    input_size = len(dataset[0][0])
    if input_size != network_specification[0]:
        raise InvalidDimensionError


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


def _collection_params(hidden_layers, output_layer):
    params = []
    for layer in hidden_layers:
        params = params + layer.params
    params = params + output_layer.params
    return params


class MultiLayerPerceptronRegressor(object):
    def __init__(self, seed, dataset, network_specification):
        # Assert inputs
        _verify_network_specification(network_specification)
        _verify_dimensions(dataset, network_specification)

        # Prepare data
        self._data_x, self._data_y, self._data_points = _prepare_data(dataset)

        # Build network
        self._input_layer = tensor.matrix('x')
        self._hidden_layers = _add_hidden_layers(seed, self._input_layer, network_specification)
        self._output_layer = _add_linear_regression_layer(seed, self._hidden_layers, network_specification)
        self._output_vector = tensor.matrix('y')
        self._params = _collection_params(self._hidden_layers, self._output_layer)

        # Prediction function
        self._predict = theano.function(inputs=[self._input_layer],
                                        outputs=self._output_layer.predict()[0]
                                        )

    def train(self, iterations=10, learning_rate=0.1, batch_size=1):
        cost_function = self._output_layer.cost(self._output_vector)  # TODO: should this really be output vector
        gradients = [tensor.grad(cost_function, param) for param in self._params]
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self._params, gradients)]

        index = tensor.lscalar()
        train_model = theano.function(inputs=[index],
                                      outputs=cost_function,
                                      updates=updates,
                                      givens={
                                          self._input_layer: self._data_x[index * batch_size:(index + 1) * batch_size],
                                          self._output_vector: self._data_y[index * batch_size:(index + 1) * batch_size]
                                      })

        for i in range(iterations):
            for batch in range(self._data_points / batch_size):
                train_model(batch)

    def predict(self, input_vector):
        return self._predict(input_vector)


class InvalidNetworkError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Network must consist of at least an input layer, 1 hidden layer, and an output layer')


class InvalidDataError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Data set should be formatted as follow: [[[input],[output]],[[input],[output]]]')


class NoNumpyArrayError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Data should consist of numpy arrays')


class NoDatasetFoundError(Exception):
    def __init__(self):
        Exception.__init__(self, 'No input data has been found')


class InvalidDimensionError(Exception):
    def __init__(self):
        Exception.__init__(self, 'The input and output sizes of the data set do not match with the specified network')
