# In the name of God
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, number_of_clusters, max_iterations):
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations
        self.clusters = None

    def cluster(self, data):
        center_indexes = self._find_random_centers(len(data))
        distances = self._compute_distances(data)
        iteration_counter = 0

        while iteration_counter != self.max_iterations:
            iteration_counter += 1
            self.clusters = {i: [data[c]] for i, c in enumerate(center_indexes)}
            clusters_indexes = {i: [] for i in range(self.number_of_clusters)}

            self.__assign_data_to_clusters(center_indexes, clusters_indexes, data, distances)
            temp_center_indexes = self._update_centers(clusters_indexes, distances)
            if self._is_converged(temp_center_indexes, center_indexes):
                return
            center_indexes = temp_center_indexes

    def __assign_data_to_clusters(self, center_indexes, clusters_indexes, data, distances):
        for i, d in enumerate(data):
            nearest_cluster_index = -1
            nearest_center_distance = float('+inf')
            for cluster_index, center_index in enumerate(center_indexes):
                curr_distance = distances[i][center_index]
                if curr_distance < nearest_center_distance:
                    nearest_center_distance = curr_distance
                    nearest_cluster_index = cluster_index
            self.clusters[nearest_cluster_index].append(d)
            clusters_indexes[nearest_cluster_index].append(i)

    def _update_centers(self, clusters_indexes, distances):
        temp_center_indexes = list()
        for cluster in self.clusters:
            new_center_index = None
            min_distance = float('+inf')
            for j in clusters_indexes[cluster]:
                curr_distance = 0
                for k in clusters_indexes[cluster]:
                    curr_distance += distances[j][k]
                if curr_distance < min_distance:
                    min_distance = curr_distance
                    new_center_index = j
            temp_center_indexes.append(new_center_index)
        return temp_center_indexes

    def _find_random_centers(self, count_of_data_points):
        random_centers = np.random.choice([i for i in range(count_of_data_points)], size=self.number_of_clusters,
                                          replace=False)
        return random_centers

    @staticmethod
    def _compute_distances(data):
        distance_matrix = np.zeros((len(data), len(data)), dtype=float)
        for i in range(len(data)):
            for j in range(len(data)):
                dist = 0
                for k in range(data[0].shape[0]):
                    dist += (data[i][k] - data[j][k]) ** 2
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        return distance_matrix

    @staticmethod
    def _is_converged(old_centers, new_centers):
        return np.all(old_centers == new_centers)

    def plot_clusters(self):
        x = list()
        y = list()
        colors = list()
        for cluster in self.clusters:
            for data_point in self.clusters[cluster]:
                x.append(data_point[0])
                y.append(data_point[1])
                colors.append(cluster)
        plt.scatter(x, y, c=colors, alpha=0.8)
        plt.show()
