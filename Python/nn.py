import numpy as np

# Activatie functies

relu = {
    'normal': lambda x: x * (x > 0),
    'derivative': lambda x: 1 * (x > 0)
}

sigmoid = {
    'normal': lambda x: 1.0 / (1.0 + np.exp(-x)),
    'derivative': lambda x: sigmoid['normal'](x) * (1 - sigmoid['normal'](x))
}


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.lr = 0.1
        self.epochs = 10

    # Calculates a single layer.
    def __feed_forward(self, layer, input_array):
        # Input array has to have only 1 colom.
        assert input_array.shape[1] == 1
        layer['input'] = input_array
        layer['output'] = layer['activation']['normal'](
            np.dot(layer['weights'], input_array) + layer['biases'])
        return layer['output']

    def __update_layer(self, layer, errors_array, idx):
        layer['weights'] = self.lr * np.dot((errors_array * layer['activation']['derivative'](
            layer['output'])), np.transpose(layer['input']))

        layer['biases'] = layer['biases'] + errors_array * self.lr
        self.layers[idx] = layer

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

    def fit(self, input_list, targets_list):
        # Stap 1: verander de lists naar numpy arrays
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        results = inputs

        # feed-forward, zodat je de output krijgt (inputs worden ook in de layer opgeslagen)
        for layer in self.layers:
            results = self.__feed_forward(layer, results)

        # Bereken de output-error
        errors = targets - results

        # Houdt de index bij om het op te slaan in update

        # Backpropagation -> update the weights of the layer
        for t in range(self.epochs):
            idx = len(self.layers) - 1

            for layer in reversed(self.layers):
                print(f'layer: {idx}')
                print(layer['input'])
                self.__update_layer(layer, errors, idx)
                errors = np.dot(layer['weights'].T, errors)
                print('errors:', errors)
                idx = idx - 1


nn = NeuralNetwork()

nn.add_layer(2, 3, sigmoid)
nn.add_layer(3, 2)

nn.fit([4, 5], [4, 8])


print('\nPredirtion::')
pred = nn.predict([2, 3])
print(pred)
