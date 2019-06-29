# In the name of God

from collections import defaultdict
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from utils import geometric_series, generate_data_set_from_file, diamond_neighbourhood, linear_neighbourhood, \
    linear_series


class SOM:
    def __init__(self, number_of_clusters, learning_rate_generator, neighbourhood_function, number_of_dimensions=1,
                 rows=None, columns=None, max_iterations=100):
        self.weights = None
        self.number_of_clusters = number_of_clusters
        self.learning_rate_generator = learning_rate_generator
        self.neighbourhood_function = neighbourhood_function
        self.number_of_dimensions = number_of_dimensions
        self.neighbourhood_radius = self._max_count_of_neighbours(number_of_dimensions)
        self.rows = rows
        self.columns = columns
        self.max_iterations = max_iterations

    def learn(self, input_data):
        self.weights = np.random.random_sample([input_data[0].shape[0], self.number_of_clusters]).T
        iteration_counter = 0
        while True:
            iteration_counter += 1
            alpha = next(self.learning_rate_generator)
            for data in input_data:
                winner_cluster = self.find_nearest_cluster(data)
                self._update_weights(winner_cluster, data, alpha)
            if iteration_counter == self.max_iterations:
                break

    def _update_weights(self, cluster_index, data, alpha):
        if self.number_of_dimensions == 1:
            neighbours = self.neighbourhood_function(cluster_index, radius=1)
            for neighbour in neighbours:
                index = cluster_index + neighbour
                if not (0 < index < self.weights.shape[0]):
                    continue
                self._update_weight_column(index, data, alpha)
        else:
            neighbours = self.neighbourhood_function(cluster_index, self.columns,
                                                     self.rows)
            for neighbour in neighbours:
                self._update_weight_column(neighbour, data, alpha)

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

    def _update_weight_column(self, cluster_index, data, alpha):
        self.weights[cluster_index] = self.weights[cluster_index] + alpha * (data - self.weights[cluster_index])

    def _is_net_converged(self, old_weights, epsilon: float = 0):
        return np.all(np.abs(self.weights - old_weights) < epsilon)

    def _max_count_of_neighbours(self, number_of_dimensions):
        if number_of_dimensions == 1:
            return int(self.number_of_clusters / 2)
        elif number_of_dimensions == 2:
            return int(sqrt(self.number_of_clusters) / 2)
        else:
            raise Exception(f'number of dimensions must be one or two, got {number_of_dimensions}.')


def cluster_characters():
    som = SOM(21, geometric_series(.8, .05), neighbourhood_function=diamond_neighbourhood, number_of_dimensions=2,
              rows=7, columns=3)
    data, _, font_info = generate_data_set_from_file(file_path='input/hw1_input.txt', font_info=True)
    for d in data:
        d = d.reshape((d.shape[0], 1))

    som.learn(data)
    clusters = defaultdict(list)
    for i, d in enumerate(data):
        cluster = som.find_nearest_cluster(d)
        clusters[cluster].append(font_info[i])
    for c in clusters:
        print(f'cluster is {c} with {len(clusters[c])} elements')
        for d in clusters[c]:
            print(d)


def solve_tsp():
    cities = list()

    def read_input():
        with open('input/tsp_cities.in') as f:
            for l in f.readlines():
                x, y = list(map(float, l.split()))
                cities.append(np.array([x, y]))

    def find_tour():
        cx, cy, x, y = list(), list(), list(), list()
        for i, c in enumerate(cities):
            winner = net.find_nearest_cluster(c)
            x.append(net.weights[winner][0])
            y.append(net.weights[winner][1])
            cx.append(c[0])
            cy.append(c[1])
        x.append(x[0])
        y.append(y[0])
        return x, y, cx, cy

    def plot_clusters():
        x, y, cx, cy = find_tour()
        plt.figure()
        plt.scatter(cx, cy, s=120)
        plt.scatter(x, y, s=60, c='y')
        plt.plot(x, y, 'C3')
        plt.title('tsp')
        plt.show()

    read_input()

    net = SOM(number_of_clusters=len(cities) + 1,
              learning_rate_generator=geometric_series(1, 0.999),
              neighbourhood_function=linear_neighbourhood,
              max_iterations=200)

    net.learn(cities)

    plot_clusters()


if __name__ == '__main__':
    cluster_characters()
    solve_tsp()
