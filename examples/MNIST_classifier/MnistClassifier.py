import six.moves.cPickle as pickle
import gzip

import numpy

from ann.MultiLayerPerceptron import MultiLayerPerceptronClassifier


def format_data(dataset):
    return numpy.asarray([[entry[0], entry[1]] for entry in zip(dataset[0].tolist(), dataset[1])])


def load_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    training_data = format_data(train_set)
    validation_data = format_data(valid_set)
    test_data = format_data(test_set)

    return training_data, validation_data, test_data


def test_mlp(learning_rate=0.01, iterations=1, dataset='mnist.pkl.gz', batch_size=20):
    # Prepare data
    training_data, validation_data, test_data = load_data(dataset)

    # Create network
    network_specification = [784, 392, 196, 98, 49, 10]
    multilayer_perceptron_classifier = MultiLayerPerceptronClassifier(seed=1234,
                                                                      # TODO: add validation data
                                                                      network_specification=network_specification)

    # Train
    multilayer_perceptron_classifier.train(training_data, iterations=iterations, learning_rate=learning_rate, batch_size=batch_size)

    # Test
    print "Error rate of " + str(multilayer_perceptron_classifier.test(test_data)) + "%"

if __name__ == '__main__':
    test_mlp()
