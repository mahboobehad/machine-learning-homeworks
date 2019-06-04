# In the name of God
import numpy as np
from utils import binary, generate_data_set_from_file


class OneLayerPerceptronNet:
    def __init__(self, input_layer_size, output_layer_size, learning_rate, activation_function):
        self.weights = np.random.random_sample([input_layer_size, output_layer_size])
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.bias = 1

    def learn(self, input_data, data_label, epoch=100):
        for _ in range(epoch):
            for i, input_vector in enumerate(input_data):
                self._update_weights(input_vector, data_label[i])

    def _update_weights(self, input_vector, output_vector):
        y = self.activation_function(self.bias + input_vector @ self.weights)
        if y != output_vector[0]:
            for i in range(self.weights.shape[0]):
                self.weights[i] = self.weights[i] + self.learning_rate * input_vector[i] * output_vector[0]
                self.bias = self.bias + self.learning_rate * output_vector[0]

    def classify(self, input_vector):
        return self.activation_function(self.bias + input_vector @ self.weights)

    def __repr__(self):
        return '{0!s} {0!r}'.format(self.__dict__)


def solve_first_hw():
    target_chars = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
    for t in target_chars:
        print("target char is {}".format(t))
        data, label = generate_data_set_from_file('./input/hw1_input.txt', target_character=t)
        net = OneLayerPerceptronNet(data[0].shape[0], 1, .8, activation_function=binary)
        net.learn(data, label, epoch=500)
        for d in data:
            print(f'result of classification is {net.classify(d)}')


if __name__ == '__main__':
    solve_first_hw()
