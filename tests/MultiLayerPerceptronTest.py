import gzip
import unittest
import numpy as np
import six.moves.cPickle as pickle

from ann.Layers import InputLayer, LogisticRegressionLayer, LinearRegressionLayer, HiddenLayer, InvalidDimensionError, \
    LeNetConvPoolLayer
from ann.MultiLayerPerceptron import MultiLayerPerceptron, InvalidNetworkError, InvalidDataError, \
    NoDataSetFoundError, NoNumpyArrayError


def _format_data_set(data_set):
    return np.asarray([data_set[0].tolist(), data_set[1].tolist()]).T


def _load_data(data_set):
    with gzip.open(data_set, 'rb') as f:
        _, train_set, test_set = pickle.load(f)  # use validation set instead of training set to speed up test time
    return _format_data_set(train_set), _format_data_set(test_set)


class MultiLayerPerceptronTest(unittest.TestCase):
    def test_invalid_network_specification(self):
        # Given
        network_specification = [InputLayer([2]), LinearRegressionLayer(2)]

        # Then
        self.assertRaises(InvalidNetworkError, MultiLayerPerceptron,
                          seed=1234,
                          network_specification=network_specification)

    def test_invalid_data_set_format(self):
        # Given
        training_set = [[[1]]]
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=[InputLayer([1]), HiddenLayer(2),
                                                                                      LinearRegressionLayer(1)])

        # Then
        self.assertRaises(InvalidDataError, multilayer_perceptron_regressor.train, training_set)

    def test_missing_data_set(self):
        # Given
        training_set = []
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=[InputLayer([1]), HiddenLayer(2),
                                                                                      LinearRegressionLayer(1)])

        # Then
        self.assertRaises(NoDataSetFoundError, multilayer_perceptron_regressor.train, training_set)

    def test_no_numpy_array(self):
        # Given
        training_set = [[[1, 2, 3, 4], [1, 2]]]
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=[InputLayer([4]), HiddenLayer(2),
                                                                                      LinearRegressionLayer(2)])

        # Then
        self.assertRaises(NoNumpyArrayError, multilayer_perceptron_regressor.train, training_set)

    def test_invalid_input_size(self):
        # Given
        training_set = np.array([[[1, 1], [2]]])
        network_specification = [InputLayer([3]), HiddenLayer(2), LinearRegressionLayer(1)]
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=network_specification)

        # Then
        self.assertRaises(InvalidDimensionError, multilayer_perceptron_regressor.train, training_set)

    def test_invalid_output_size_regressor(self):
        # Given
        training_set = np.array([[[1, 1], [2]]])
        network_specification = [InputLayer([2]), HiddenLayer(2), LinearRegressionLayer(2)]
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=network_specification)

        # Then
        self.assertRaises(InvalidDimensionError, multilayer_perceptron_regressor.train, training_set)

    def test_invalid_output_size_classifier(self):
        # Given
        training_set = np.array([[[1, 1], [2]]])
        network_specification = [InputLayer([2]), HiddenLayer(2), LinearRegressionLayer(2)]
        multilayer_perceptron_classifier = MultiLayerPerceptron(seed=1234,
                                                                network_specification=network_specification)

        # Then
        self.assertRaises(InvalidDimensionError, multilayer_perceptron_classifier.train, training_set)

    def test_network_initialization(self):
        # Given
        network_specification = [InputLayer([4]), HiddenLayer(3), LinearRegressionLayer(2)]

        # When
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=network_specification)

        # Then
        self.assertEqual(1, len(multilayer_perceptron_regressor._network_specification[1:-1]))
        self.assertEqual(3, multilayer_perceptron_regressor._network_specification[1].size)
        self.assertEqual(2, multilayer_perceptron_regressor._output_layer.size)

    def test_XOR_problem_regression(self):
        # Given
        network_specification = [InputLayer([2]), HiddenLayer(2), LinearRegressionLayer(1)]
        training_set = np.asarray([[[0.0, 0.0], [0.0]],
                                   [[0.0, 1.0], [1.0]],
                                   [[1.0, 0.0], [1.0]],
                                   [[1.0, 1.0], [0.0]]
                                   ])

        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234,
                                                               network_specification=network_specification)

        # When
        multilayer_perceptron_regressor.train(training_set, iterations=1000, learning_rate=0.1)

        # Then
        self.assertTrue(multilayer_perceptron_regressor.predict([[0, 0]])[0] < 0.0001)
        self.assertTrue(multilayer_perceptron_regressor.predict([[0, 1]])[0] > 0.9999)
        self.assertTrue(multilayer_perceptron_regressor.predict([[1, 0]])[0] > 0.9999)
        self.assertTrue(multilayer_perceptron_regressor.predict([[1, 1]])[0] < 0.0001)

        self.assertTrue(multilayer_perceptron_regressor.test(training_set) < 0.0001)

    def test_XOR_problem_classification(self):
        # Given
        network_specification = [InputLayer([2]), HiddenLayer(4), LogisticRegressionLayer(2)]
        training_set = np.asarray([[[0.0, 0.0], 0],
                                   [[0.0, 1.0], 1],
                                   [[1.0, 0.0], 1],
                                   [[1.0, 1.0], 0]
                                   ])

        multilayer_perceptron_classifier = MultiLayerPerceptron(seed=1234,
                                                                network_specification=network_specification)

        # When
        multilayer_perceptron_classifier.train(training_set, iterations=100, learning_rate=0.1)

        # Then
        self.assertEqual(0, multilayer_perceptron_classifier.predict([[0.0, 0.0]]))
        self.assertEqual(1, multilayer_perceptron_classifier.predict([[0.0, 1.0]]))
        self.assertEqual(1, multilayer_perceptron_classifier.predict([[1.0, 0.0]]))
        self.assertEqual(0, multilayer_perceptron_classifier.predict([[0.0, 0.0]]))

        self.assertTrue(multilayer_perceptron_classifier.test(training_set) == 0)

    def test_mnist_classifier(self):
        # Given
        training_set, test_set = _load_data('../data/mnist.pkl.gz')

        network_specification = [InputLayer([28, 28]),
                                 LeNetConvPoolLayer(feature_map=2, filter_shape=(5, 5), pool_size=(2, 2)),
                                 HiddenLayer(50),
                                 LogisticRegressionLayer(10)]
        neural_network = MultiLayerPerceptron(seed=1234, network_specification=network_specification)

        # When
        neural_network.train(training_set=training_set, learning_rate=0.1, batch_size=500, iterations=1)

        # Then
        self.assertEqual(28.18, round(neural_network.test(test_set=test_set, batch_size=1000), 2))
