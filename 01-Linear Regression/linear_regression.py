import numpy as np
from utils.features import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
            1. data preprocess
            2. get the number of features
            3. initialize
        """
        data_processed, features_mean, features_deviation = \
            prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # features: x(1), x(2), x(3)...
        num_features = self.data.shape[1]
        # Θ matrix
        self.theta = np.zeros((num_features, 1))

    def train(self, learning_rate, num_iteration=500):
        """
            do gradient descent
        """
        losses = self.gradient_descent(learning_rate, num_iteration)

        return self.theta, losses

    def gradient_descent(self, learning_rate, num_iteration=500):
        # stores all the losses from each iteration
        losses = []
        for _ in range(num_iteration):
            self.gradient_step(learning_rate)
            losses.append(self.loss_func(self.data, self.labels))

        return losses

    def gradient_step(self, learning_rate):
        num_samples = self.data.shape[0]
        pred = LinearRegression.make_prediction(self.data, self.theta)
        # error: [num_samples, 1]
        error = pred - self.labels
        theta = self.theta
        # update theta using gradient descent
        # error.T: [1, num_sampels]
        # data: [num_samples, num_features]
        # theta: [num_samples, 1]
        # to update theta, should deduct a matrix with shape of [num_samples, 1]
        theta = theta - learning_rate * (1 / num_samples) * (np.dot(error.T, self.data)).T
        self.theta = theta


    def loss_func(self, data, labels):
        num_samples = data.shape[0]
        pred = LinearRegression.make_prediction(self.data, self.theta)
        # error: [num_samples, 1]
        error = pred - self.labels
        # loss: [1, 1]
        loss = 1 / 2 * np.dot(error.T, error) / num_samples

        return loss[0][0]

    @staticmethod
    def make_prediction(data, theta):
        # data shape: [num_sample, num_feature]
        # theta shape: [num_features, 1]
        pred = np.dot(data, theta)

        return pred

    def get_loss(self, data, labels):
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        return self.loss_func(data_processed, labels)

    def predict(self, data):
        """
            predict after training
        """
        data_processed = \
            prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]

        return LinearRegression.make_prediction(data_processed, self.theta)




