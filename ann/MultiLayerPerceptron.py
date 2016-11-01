import numpy as np
import theano
from theano import tensor as tensor


def _add_hidden_layers(seed, network_specification):
    hidden_layers = []
    for index, layer in enumerate(network_specification):
        if 0 == index or index == len(network_specification) - 1:
            continue

        network_specification[index].connect(seed=seed, input_layer=network_specification[index - 1])
        hidden_layers.append(network_specification[index])
    return hidden_layers


def _verify_network_specification(network_specification):
    if len(network_specification) < 3:
        raise InvalidNetworkError


def _verify_data_set(data_set):
    if len(data_set) < 1:
        raise NoDatasetFoundError
    if len(data_set[0]) != 2:
        raise InvalidDataError


def _collection_params(hidden_layers, output_layer):
    params = []
    for layer in hidden_layers:
        params = params + layer.params
    params = params + output_layer.params
    return params


def _prepare_data(data_set):
    try:
        data_set = data_set.T

        # TODO: I need to convert tolist before to array because lowest level is list?
        data_x = np.asarray(data_set[0].tolist())
        data_y = np.asarray(data_set[1].tolist())

        shared_x = theano.shared(data_x, borrow=True)
        shared_y = theano.shared(data_y, borrow=True)

        return shared_x, shared_y, len(data_x)
    except AttributeError:
        raise NoNumpyArrayError


def _verify_and_prepare_data(data_set, network_specification):
    _verify_data_set(data_set)
    network_specification[0].verify_dimensions(data_set)
    network_specification[-1].verify_dimensions(data_set)
    return _prepare_data(data_set)


class MultiLayerPerceptron(object):
    def __init__(self, seed, network_specification):
        # Assert network specification
        _verify_network_specification(network_specification)

        # Build network
        self.network_specification = network_specification
        self._input_vector = network_specification[0].output_stream
        self._hidden_layers = _add_hidden_layers(seed, network_specification)
        self._output_layer = self._add_output_layer(seed, self._hidden_layers)
        self._output_vector = network_specification[-1].output_vector
        self._params = _collection_params(self._hidden_layers, self._output_layer)

        # Prediction function
        self._predict = theano.function(inputs=[self._input_vector],
                                        outputs=self._output_layer.predict()
                                        )

    def train(self, training_set, iterations=10, learning_rate=0.1, batch_size=1):
        data_x, data_y, data_points = _verify_and_prepare_data(training_set, self.network_specification)

        cost_function = self._output_layer.cost(self._output_vector)
        gradients = [tensor.grad(cost_function, param) for param in self._params]
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self._params, gradients)]
        index = tensor.lscalar()
        train_model = theano.function(inputs=[index],
                                      outputs=cost_function,
                                      updates=updates,
                                      givens={
                                          self._input_vector: data_x[index * batch_size:(index + 1) * batch_size],
                                          self._output_vector: data_y[index * batch_size:(index + 1) * batch_size]
                                      })

        for i in range(iterations):
            if i % 10 == 0:
                print "iteration " + str(i + 1) + "/" + str(iterations)
            for batch in range(data_points / batch_size):
                train_model(batch)

    def test(self, test_set, batch_size=1):
        data_x, data_y, _ = _verify_and_prepare_data(test_set, self.network_specification)

        index = tensor.lscalar()
        test_model = theano.function(inputs=[index],
                                     outputs=self._output_layer.error(self._output_vector),
                                     givens={
                                         self._input_vector: data_x[index * batch_size:(index + 1) * batch_size],
                                         self._output_vector: data_y[index * batch_size:(index + 1) * batch_size]
                                     })

        test_loss = [test_model(i) for i in range(len(test_set) / batch_size)]
        return np.mean(test_loss) * 100

    def predict(self, input_vector):
        return self._predict(input_vector)

    def _add_output_layer(self, seed, hidden_layers):
        self.network_specification[-1].connect(seed=seed,
                                               input_layer=hidden_layers[-1]
                                               )
        return self.network_specification[-1]


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
