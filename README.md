# Ann: simple construction of artificial neural networks
A toy project that simplifies construction of neural networks by allowing a user to specify layers in a list. Hyperparameters can be passed in to the specific layers or to the train method.

## Example use
```python
# Create network
network_specification = [InputLayer([28, 28]),
                         LeNetConvPoolLayer(feature_map=20, filter_shape=(5, 5), pool_size=(2, 2)),
                         LeNetConvPoolLayer(feature_map=50, filter_shape=(5, 5), pool_size=(2, 2)),
                         HiddenLayer(120),
                         HiddenLayer(84),
                         LogisticRegressionLayer(10)]

neural_network = MultiLayerPerceptron(seed=1234, network_specification=network_specification)

# Train
neural_network.train(training_set=training_set, learning_rate=0.1, batch_size=500, iterations=50)
```

## Features
Features include, but are not limited to:
- Hidden layers
- Convolutional layers
- Classification
- Multi variable regression
- GPU enabled