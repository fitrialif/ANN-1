import unittest
import numpy as np

from ann.Layers import InputLayer, LogisticRegressionLayer, LinearRegressionLayer, HiddenLayer, InvalidDimensionError
from ann.MultiLayerPerceptron import MultiLayerPerceptron, InvalidNetworkError, InvalidDataError, \
    NoDatasetFoundError, NoNumpyArrayError


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
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234, network_specification=[InputLayer([1]), HiddenLayer(2), LinearRegressionLayer(1)])

        # Then
        self.assertRaises(InvalidDataError, multilayer_perceptron_regressor.train, training_set)

    def test_missing_data_set(self):
        # Given
        training_set = []
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234, network_specification=[InputLayer([1]), HiddenLayer(2), LinearRegressionLayer(1)])

        # Then
        self.assertRaises(NoDatasetFoundError, multilayer_perceptron_regressor.train, training_set)

    def test_no_numpy_array(self):
        # Given
        training_set = [[[1, 2, 3, 4], [1, 2]]]
        multilayer_perceptron_regressor = MultiLayerPerceptron(seed=1234, network_specification=[InputLayer([4]), HiddenLayer(2), LinearRegressionLayer(2)])

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
