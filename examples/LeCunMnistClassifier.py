import six.moves.cPickle as pickle
import gzip
import numpy as np

from ann.Layers import LeNetConvPoolLayer, InputLayer, HiddenLayer, LogisticRegressionLayer
from ann.MultiLayerPerceptron import MultiLayerPerceptron


def format_data_set(data_set):
    return np.asarray([data_set[0].tolist(), data_set[1].tolist()]).T


def load_data(data_set):
    with gzip.open(data_set, 'rb') as f:
        train_set, _, test_set = pickle.load(f)
    return format_data_set(train_set), format_data_set(test_set)


def le_cun_minst_classifier(data_set='../data/mnist.pkl.gz'):
    # Prepare data
    training_set, test_set = load_data(data_set)

    # Create network
    network_specification = [InputLayer([28, 28]),
                             LeNetConvPoolLayer(feature_map=20, filter_shape=(5, 5), pool_size=(2, 2)),
                             LeNetConvPoolLayer(feature_map=50, filter_shape=(5, 5), pool_size=(2, 2)),
                             HiddenLayer(500),
                             LogisticRegressionLayer(10)]

    neural_network = MultiLayerPerceptron(seed=1234, network_specification=network_specification)

    # Train
    neural_network.train(training_set=training_set, learning_rate=0.1, batch_size=100, iterations=50)

    # Test
    print "Error rate of {}%".format(neural_network.test(test_set=test_set, batch_size=1000))


if __name__ == '__main__':
    le_cun_minst_classifier()
