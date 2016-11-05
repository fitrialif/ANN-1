import unittest
from ann.Layers import HiddenLayer, LinearRegressionLayer, LogisticRegressionLayer, InputLayer, LeNetConvPoolLayer


class HiddenLayerTest(unittest.TestCase):
    def test_initialization(self):
        # Given, a total of 6 in and out will result into weights between -1 and 1 according to xavier method
        input_layer = InputLayer([3])
        hidden_layer = HiddenLayer(3)

        # When
        hidden_layer.init_weights(seed=1234, input_layer=input_layer)

        # Then
        params = hidden_layer.params
        weights = params[0].eval()
        for column in weights:
            for value in column:
                self.assertTrue(-1 <= value <= 1)

        bias = params[1].eval()
        for b in bias:
            self.assertEqual(b, 0)


class LinearRegressionLayerTest(unittest.TestCase):
    def test_initialization(self):
        # Given, a total of 6 in and out will result into weights between -1 and 1 according to xavier method
        hidden_layer = HiddenLayer(3)
        linear_regression_layer = LinearRegressionLayer(3)

        # When
        linear_regression_layer.init_weights(seed=1234, input_layer=hidden_layer)

        # Then
        params = linear_regression_layer.params
        weights = params[0].eval()
        for column in weights:
            for value in column:
                self.assertTrue(-1 <= value <= 1)

        bias = params[1].eval()
        for b in bias:
            self.assertEqual(b, 0)


class LogisticRegressionTest(unittest.TestCase):
    def test_initialization(self):
        # Given, a total of 6 in and out will result into weights between -1 and 1 according to xavier method
        hidden_layer = HiddenLayer(3)
        linear_regression_layer = LogisticRegressionLayer(3)

        # When
        linear_regression_layer.init_weights(seed=1234, input_layer=hidden_layer)

        # Then
        params = linear_regression_layer.params
        weights = params[0].eval()
        for column in weights:
            for value in column:
                self.assertTrue(-1 <= value <= 1)

        bias = params[1].eval()
        for b in bias:
            self.assertEqual(b, 0)


class LeNetConvPoolLayerTest(unittest.TestCase):
    def test_initialization(self):
        # Given
        input_layer = InputLayer([4, 4])
        conv_layer = LeNetConvPoolLayer(feature_map=1, filter_shape=(2, 2), pool_size=(1, 2))

        # When
        conv_layer.init_weights(seed=1234, input_layer=input_layer)

        # Then
        params = conv_layer.params
        weights_matrix = params[0].eval()
        for dim in weights_matrix:
            for feature_map in dim:
                for column in feature_map:
                    for value in column:
                        self.assertTrue(-1 <= value <= 1)

        bias = params[1].eval()
        for b in bias:
            self.assertEqual(b, 0)

        self.assertEqual((3, 1), conv_layer.output_shape)
        self.assertEqual(3, conv_layer.size)
