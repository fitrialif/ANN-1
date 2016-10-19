from ann.MultiLayerPerceptron import MultiLayerPerceptronClassifier
import numpy as np


def xor_regression():
    network_specification = [2, 4, 2]
    training_set = np.asarray([[[0.0, 0.0], 0],
                               [[0.0, 1.0], 1],
                               [[1.0, 0.0], 1],
                               [[1.0, 1.0], 0]
                               ])

    multilayer_perceptron_regressor = MultiLayerPerceptronClassifier(seed=1234,
                                                                     network_specification=network_specification)

    multilayer_perceptron_regressor.train(training_set, iterations=1000, learning_rate=0.1)

    print "0:0 leads to: " + str(multilayer_perceptron_regressor.predict([[0, 0]])[0])
    print "1:0 leads to: " + str(multilayer_perceptron_regressor.predict([[1, 0]])[0])
    print "0:1 leads to: " + str(multilayer_perceptron_regressor.predict([[0, 1]])[0])
    print "1:1 leads to: " + str(multilayer_perceptron_regressor.predict([[1, 1]])[0])


if __name__ == '__main__':
    xor_regression()
