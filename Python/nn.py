import numpy as np
import math

# Activatie functies


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Afgeleide van activatiefuncties


def derivative_relu(x):
    return 1 * (x > 0)


def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    # Calculates a single layer.
    def __feed_forward(self, layer, input_array):
        # Input array has to have only 1 colom.
        assert input_array.shape[1] == 1
        return layer['activation'](np.dot(layer['weights'], input_array) + layer['biases'])

    # Add a layer with default relu activation.
    def add_layer(self, input_nodes, layer_nodes, activation=relu):
        # note the order of nodes.
        weights = np.random.normal(0.0, 1.0, (layer_nodes, input_nodes))
        biases = np.random.normal(size=(layer_nodes, 1))
        self.layers.append(
            {'weights': weights, 'biases': biases, 'activation': activation})

    def predict(self, input_list):
        input_array = np.array(input_list, ndmin=2).T
        for layer in self.layers:
            input_array = self.__feed_forward(layer, input_array)
        return input_array

    def fit(self, input_list, target):
        # Step 1: Get the guesses
        target = np.array(target, ndmin=2).T
        outputs = self.predict(input_list)

        # Step 2: Calculate the output errors
        output_errors = target - outputs

        # Step 3: Calculate hidden layer errors
        for layer in reversed(self.layers):
            weights = np.transpose(layer['weights'])

            print('weights:')
            print(weights)

            print('\noutput_error')
            print(output_errors)

            hidden_error = np.dot(weights, output_errors)
            print('\nhidden_error:')
            print(hidden_error)
            print('\n\n')
            output_errors = hidden_error


nn = NeuralNetwork()

nn.add_layer(2, 3)
nn.add_layer(3, 2)

# model.predict([1, 1])

nn.fit([4, 9], [4, 12])
