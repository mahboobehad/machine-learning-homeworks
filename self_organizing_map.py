# In the name of God
from collections import defaultdict
from math import sqrt

import numpy as np

from utils import geometric_series, generate_data_set_from_file, linear_neighbourhood, data_to_map


class SOM:
    def __init__(self, number_of_clusters, learning_rate_generator, neighbourhood_function):
        self.weights = None
        self.number_of_clusters = number_of_clusters
        self.learning_rate_generator = learning_rate_generator
        self.neighbourhood_function = neighbourhood_function
        self.neighbourhood_radius = self._max_count_of_neighbours()

    def learn(self, input_data):
        self.weights = np.random.random_sample([input_data[0].shape[0], self.number_of_clusters]).T
        iteration_counter = 0
        while True:
            iteration_counter += 1
            print(f"iteration -> {iteration_counter}")
            old_weight = np.copy(self.weights)
            for data in input_data:
                cluster_index = self.find_nearest_cluster(data)
                self._update_weights(cluster_index, data)

            if self._is_net_converged(old_weight, epsilon=.00001) and iteration_counter == 100:
                print(f"converged")
                break

    def _update_weights(self, cluster_index, data):
        print(f'max number of neighbours is {self.max_neighbourhood_radius}')
        for neighbour in self.neighbourhood_function(cluster_index, radius=self.max_neighbourhood_radius):
            index = cluster_index + neighbour
            if not (0 < index < self.weights.shape[0]):
                break
            self._update_weight_column(index, data)

    def find_nearest_cluster(self, data):
        cluster_index = None
        min_distance = float('inf')
        for j in range(self.number_of_clusters):
            distance = self._compute_distance(data, j)
            if distance < min_distance:
                min_distance = distance
                cluster_index = j
        return cluster_index

    def _compute_distance(self, data, cluster_index):
        distance = np.sum((data - self.weights[cluster_index]) ** 2)
        return distance

    def _update_weight_column(self, cluster_index, data):
        alpha = next(self.learning_rate_generator)
        self.weights[cluster_index] = self.weights[cluster_index] + alpha * (data - self.weights[cluster_index])

    def _is_net_converged(self, old_weights, epsilon=0):
        return not np.any(np.abs(self.weights - old_weights) > epsilon)

    def _max_count_of_neighbours(self, number_of_dimensions):
        if number_of_dimensions == 1:
            return int(self.number_of_clusters / 2)
        elif number_of_dimensions == 2:
            return int(sqrt(self.number_of_clusters) / 2)
        else:
            raise Exception(f'number of dimensions must be one or two, got {number_of_dimensions}.')


if __name__ == '__main__':
    som = SOM(7, geometric_series(a_0=.6, q=.25), neighbourhood_function=linear_neighbourhood)
    data, _ = generate_data_set_from_file(file_path='input/hw1_input.txt')
    for d in data:
        d = d.reshape((d.shape[0], 1))

    som.learn(data)
    clusters = defaultdict(list)
    for i, d in enumerate(data):
        cluster = som.find_nearest_cluster(d)
        clusters[chr(ord('A') + cluster)].append(i)
    print(clusters.keys())
