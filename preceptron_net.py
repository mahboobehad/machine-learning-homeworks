# In the name of God
import numpy as np


class PerceptronNet:
    def __init__(self, input_layer, output_layer, learning_rate, activation_function):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.weights = np.zeros([input_layer.shape[0], output_layer.shape[0]])
        self.learning_rate = learning_rate
        self.activation_function = activation_function

    def _update_weights(self, input_vector, output_vector):
        y = self.activation_function(input_vector @ self.weights)
        if y != output_vector:
            for i in range(self.weights.shape[0]):
                self.weights[i] = self.weights[i] + self.learning_rate * input_vector[i]

    def learn(self, input_data, targets):
        for i, input_vector in enumerate(input_data):
            self._update_weights(input_data, targets[i])

    def __repr__(self):
        return '{0!s} {0!r}'.format(self.__dict__)


def ac_func(x):
    return -1 if x < .5 else 0 if -.5 <= x <= .5 else 1


def read_input_file():
    pass


if __name__ == '__main__':
    net = PerceptronNet(np.zeros([64]), np.zeros([1]), .5, activation_function=ac_func)
