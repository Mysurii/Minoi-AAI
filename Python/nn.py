import numpy as np

# Activation functions


def relu(x):
    return x * (x > 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.e**-x)

# Derivative of activation functions


def derivative_relu(x):
    return 1 * (x > 0)


def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    # Add a layer with default relu activation.
    def add_layer(self, input_nodes, layer_nodes, activation=relu):
        # note the order of nodes.
        weights = np.random.normal(0.0, 1.0, (layer_nodes, input_nodes))
        biases = np.random.normal(size=(layer_nodes, 1))
        self.layers.append(
            {'weights': weights, 'biases': biases, 'activation': activation})

    # Calculates a single layer.
    def __feed_forward(self, layer, input_array):
        # Input array has to have only 1 colom.
        assert input_array.shape[1] == 1
        return layer['activation'](np.dot(layer['weights'], input_array) + layer['biases'])

    def predict(self, input_list):
        input_array = np.array(input_list, ndmin=2).T
        for lay in self.layers:
            input_array = self.__feed_forward(lay, input_array)
        return input_array

    def fit(self, input_list, target):
        # Stap 1: Get the guesses
        outputs = self.predict(input_list)

        errors = target - outputs
        print('errors', errors)

        # Stap 2: Calculate the output errors

        print('output:', outputs)

        # target_array = np.array(target)

        # error = np.subtract(outputs, target_array)

        # print('target', target_array)
        # print('output', outputs)

        # print(error)


nn = NeuralNetwork()

nn.add_layer(2, 3, sigmoid)
nn.add_layer(3, 2)

print(nn.predict([3, 1]))

# nn.fit([12, 1], [12, 20])
