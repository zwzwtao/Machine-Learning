import numpy as np


class KMeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        # 1. randomly pick K(represented as num_clusters) centroids
        centroids = KMeans.centroids_init(self.data, self.num_clusters)
        # 2. start training
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 3. obtain all the distances from each samples to each centroid, and find the closest centroid
            # closest_centroids_ids stores all the closest centroids for each sample
            closest_centroids_ids = KMeans.find_closest_centroids(self.data, centroids)
            # 4. update centroids
            centroids = KMeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)

        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        num_examples = data.shape[0]
        # randomly pick examples(shuffle)
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clusters], :]

        return centroids

    @staticmethod
    def find_closest_centroids(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        # closest_centroids_ids stores all the closest centroids for each sample,
        # thus the total number for it is 'num_examples'
        closest_centroids_ids = np.zeros((num_examples, 1))
        for sample_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))
            for centroid_index in range(num_centroids):
                # the difference between a sample and a centroid
                distance_diff = data[sample_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff ** 2)
            closest_centroids_ids[sample_index] = np.argmin(distance)

        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        num_features = data.shape[1]
        # since we want to compute the mean value of all the feature values
        # the dimension should be (num_clusters, num_features)
        centroids = np.zeros((num_clusters, num_features))
        for centroid_id in range(num_clusters):
            closest_ids = closest_centroids_ids == centroid_id
            # update new centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)

        return centroids
