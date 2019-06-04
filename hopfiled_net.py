# In the name of God
import numpy as np

from utils import generate_data_set_from_file, bipolar


class HopfieldNet:
    def __init__(self):
        self.weights = None
        self.units = None
        self.update_order = None
        self.activation_function = bipolar

    def learn(self, patterns):
        self.units, self.update_order = self._initialize_units(patterns)
        self.update_weights(patterns)

        for p in patterns:
            self._update_units(p, self.update_order)
            print(self)

    def update_weights(self, patterns):
        self.weights = np.zeros([patterns[0].shape[0], patterns[0].shape[0]])
        for p in patterns:
            p = p.reshape((p.shape[0], 1))
            self.weights = self.weights + (p @ p.T)
        self.weights = self.weights - (len(patterns) * np.identity(patterns[0].shape[0]))
        print(f"weights are \n {self.weights}")

    @staticmethod
    def _initialize_units(patterns):
        units = np.zeros([patterns[0].shape[0]], dtype=int)
        update_order = [_ for _ in range(patterns[0].shape[0])]
        np.random.shuffle(update_order)
        return units, update_order

    def _test_convergence(self):
        pass

    def _update_units(self, pattern, units_order):
        for i in range(self.units.shape[0]):
            self.units[i] = pattern[i]
        for u in units_order:
            for j in range(self.weights.shape[0]):
                self.units[u] += (self.weights[u][j] * self.units[j])
            self.units[u] = self.activation_function(self.units[u], 0)

    def __repr__(self):
        units_repr = str()
        for i in range(self.units.shape[0]):
            if self.units[i] == -1:
                units_repr += '. '
            else:
                units_repr += '# '
            if (i + 1) % 7 == 0:
                units_repr += '\n'
        return units_repr.lstrip()


if __name__ == '__main__':
    net = HopfieldNet()
    data, _ = generate_data_set_from_file('input/hw1_input.txt', input_map=lambda x: -1 if x == '.' else 1)
    char_count = 7
    net.learn(data[0:3])
