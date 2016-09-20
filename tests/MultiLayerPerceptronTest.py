import unittest
import numpy as np

from ann.MultiLayerPerceptron import MultiLayerPerceptronRegressor, InvalidNetworkError, InvalidDataError, \
    InvalidDimensionError, NoDatasetFoundError, NoNumpyArrayError


class MultiLayerPerceptronTest(unittest.TestCase):
    def test_invalid_network_specification(self):
        # Given
        network_specification = [2, 2]

        # Then
        self.assertRaises(InvalidNetworkError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          dataset=[[[1, 1], [2]]],
                          network_specification=network_specification)

    def test_invalid_data_set_format(self):
        # Given
        dataset = [[[1]]]

        # Then
        self.assertRaises(InvalidDataError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          dataset=dataset,
                          network_specification=[1, 2, 1])

    def test_missing_data_set(self):
        # Given
        dataset = []

        # Then
        self.assertRaises(NoDatasetFoundError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          dataset=dataset,
                          network_specification=[1, 2, 3])

    def test_invalid_input_size(self):
        # Given
        dataset = [[[1, 1], [2]]]
        network_specification = [3, 2, 1]

        # Then
        self.assertRaises(InvalidDimensionError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          dataset=dataset,
                          network_specification=network_specification)

    def test_no_numpy_array(self):
        # Given
        dataset = [[[1, 2, 3, 4], [1, 2]]]

        # Then
        self.assertRaises(NoNumpyArrayError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          dataset=dataset,
                          network_specification=[4, 3, 2])

    def test_network_initialization(self):
        # Given
        network_specification = [4, 3, 2]

        # When
        multilayer_perceptron_regressor = MultiLayerPerceptronRegressor(seed=1234,
                                                                        dataset=np.array([[[1, 2, 3, 4], [1, 2]]]),
                                                                        network_specification=network_specification)

        # Then
        self.assertEqual(1, len(multilayer_perceptron_regressor._hidden_layers))
        self.assertEqual(4, multilayer_perceptron_regressor._hidden_layers[0].n_in)
        self.assertEqual(3, multilayer_perceptron_regressor._hidden_layers[0].n_out)

        self.assertEqual(3, multilayer_perceptron_regressor._output_layer.n_in)
        self.assertEqual(2, multilayer_perceptron_regressor._output_layer.n_out)

    def test_XOR_problem_regression(self):
        # Given
        network_specification = [2, 2, 1]
        dataset = np.asarray([[[1.0, 1.0], [0.0]],
                              [[0.0, 0.0], [0.0]],
                              [[1.0, 0.0], [1.0]],
                              [[0.0, 1.0], [1.0]]
                              ])

        multilayer_perceptron_regressor = MultiLayerPerceptronRegressor(seed=1234,
                                                                        dataset=dataset,
                                                                        network_specification=network_specification)

        # When
        multilayer_perceptron_regressor.train(iterations=1000, learning_rate=0.1)

        # Then
        self.assertTrue(multilayer_perceptron_regressor.predict([[0, 0]])[0] < 0.0001)
        self.assertTrue(multilayer_perceptron_regressor.predict([[0, 1]])[0] > 0.9999)
        self.assertTrue(multilayer_perceptron_regressor.predict([[1, 0]])[0] > 0.9999)
        self.assertTrue(multilayer_perceptron_regressor.predict([[1, 1]])[0] < 0.0001)