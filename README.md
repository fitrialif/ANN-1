# ANN: simple construction of artificial neural networks
A toy project that simplifies construction of neural networks by allowing a user to specify layers in a list. Hyperparameters can be passed in to the specific layers or to the train method. This is essentially a very simplified version of the Keras library, before I knenw it existed.

## Example use
We can construct a convolution network used to classify the Mnist data set in just a couple of lines:
```python
# Specify network
network_specification = [InputLayer([28, 28]),
                         LeNetConvPoolLayer(feature_map=20, filter_shape=(5, 5), pool_size=(2, 2)),
                         LeNetConvPoolLayer(feature_map=50, filter_shape=(5, 5), pool_size=(2, 2)),
                         HiddenLayer(500),
                         LogisticRegressionLayer(10)]

# Construct network
neural_network = MultiLayerPerceptron(seed=1234, network_specification=network_specification)

# Train
neural_network.train(training_set=training_set, learning_rate=0.1, batch_size=100, iterations=50)

# Test
print "Error rate of {}%".format(neural_network.test(test_set=test_set)
```

With this simple network, we end up with an accuracy of 99.15% in just 50 iterations.
