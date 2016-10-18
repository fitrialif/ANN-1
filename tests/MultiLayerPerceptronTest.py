import unittest
import numpy as np

from ann.MultiLayerPerceptron import MultiLayerPerceptronRegressor, InvalidNetworkError, InvalidDataError, \
    InvalidDimensionError, NoDatasetFoundError, NoNumpyArrayError, MultiLayerPerceptronClassifier


class MultiLayerPerceptronTest(unittest.TestCase):
    def test_invalid_network_specification(self):
        # Given
        network_specification = [2, 2]

        # Then
        self.assertRaises(InvalidNetworkError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          training_set=[[[1, 1], [2]]],
                          network_specification=network_specification)

    def test_invalid_data_set_format(self):
        # Given
        training_set = [[[1]]]

        # Then
        self.assertRaises(InvalidDataError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          training_set=training_set,
                          network_specification=[1, 2, 1])

    def test_missing_data_set(self):
        # Given
        training_set = []

        # Then
        self.assertRaises(NoDatasetFoundError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          training_set=training_set,
                          network_specification=[1, 2, 3])

    def test_no_numpy_array(self):
        # Given
        training_set = [[[1, 2, 3, 4], [1, 2]]]

        # Then
        self.assertRaises(NoNumpyArrayError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          training_set=training_set,
                          network_specification=[4, 3, 2])

    def test_invalid_input_size(self):
        # Given
        training_set = np.array([[[1, 1], [2]]])
        network_specification = [3, 2, 1]

        # Then
        self.assertRaises(InvalidDimensionError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          training_set=training_set,
                          network_specification=network_specification)

    def test_invalid_output_size_regressor(self):
        # Given
        training_set = np.array([[[1, 1], [2]]])
        network_specification = [2, 2, 2]

        # Then
        self.assertRaises(InvalidDimensionError, MultiLayerPerceptronRegressor,
                          seed=1234,
                          training_set=training_set,
                          network_specification=network_specification)

    def test_invalid_output_size_classifier(self):
        # Given
        training_set = np.array([[[1, 1], [2]]])
        network_specification = [2, 2, 2]

        # Then
        self.assertRaises(InvalidDimensionError, MultiLayerPerceptronClassifier,
                          seed=1234,
                          training_set=training_set,
                          network_specification=network_specification)

    def test_network_initialization(self):
        # Given
        network_specification = [4, 3, 2]

        # When
        multilayer_perceptron_regressor = MultiLayerPerceptronRegressor(seed=1234,
                                                                        training_set=np.array([[[1, 2, 3, 4], [1, 2]]]),
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
        training_set = np.asarray([[[0.0, 0.0], [0.0]],
                              [[0.0, 1.0], [1.0]],
                              [[1.0, 0.0], [1.0]],
                              [[1.0, 1.0], [0.0]]
                              ])

        multilayer_perceptron_regressor = MultiLayerPerceptronRegressor(seed=1234,
                                                                        training_set=training_set,
                                                                        network_specification=network_specification)

        # When
        multilayer_perceptron_regressor.train(iterations=1000, learning_rate=0.1)

        # Then
        self.assertTrue(multilayer_perceptron_regressor.predict([[0, 0]])[0] < 0.0001)
        self.assertTrue(multilayer_perceptron_regressor.predict([[0, 1]])[0] > 0.9999)
        self.assertTrue(multilayer_perceptron_regressor.predict([[1, 0]])[0] > 0.9999)
        self.assertTrue(multilayer_perceptron_regressor.predict([[1, 1]])[0] < 0.0001)

        self.assertTrue(multilayer_perceptron_regressor.test(training_set) < 0.0001)

    def test_XOR_problem_classification(self):
        # Given
        network_specification = [2, 4, 2]
        training_set = np.asarray([[[0.0, 0.0], 0],
                              [[0.0, 1.0], 1],
                              [[1.0, 0.0], 1],
                              [[1.0, 1.0], 0]
                              ])

        multilayer_perceptron_classifier = MultiLayerPerceptronClassifier(seed=1234,
                                                                          training_set=training_set,
                                                                          network_specification=network_specification)

        # When
        multilayer_perceptron_classifier.train(iterations=100, learning_rate=0.1)

        # Then
        self.assertEqual(0, multilayer_perceptron_classifier.predict([[0.0, 0.0]]))
        self.assertEqual(1, multilayer_perceptron_classifier.predict([[0.0, 1.0]]))
        self.assertEqual(1, multilayer_perceptron_classifier.predict([[1.0, 0.0]]))
        self.assertEqual(0, multilayer_perceptron_classifier.predict([[0.0, 0.0]]))

        self.assertTrue(multilayer_perceptron_classifier.test(training_set) == 0)
