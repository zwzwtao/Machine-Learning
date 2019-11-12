import numpy as np
from scipy.optimize import minimize
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid

class LogisticRegression:
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
        # multi-class will have unique_labels classes
        # thus have
        self.unique_labels = np.unique(labels)
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # features: x(1), x(2), x(3)...
        num_features = self.data.shape[1]
        # np.unique(labels).shape: (num_labels,)
        num_unique_labels = np.unique(labels).shape[0]
        # Θ matrix, since we have multiple classes, the first dimension is num_unique_labels
        self.theta = np.zeros((num_unique_labels, num_features))

    def train(self, max_iterations=1000):
        # store all the losses
        cost_histories = []
        num_features = self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            # only pick Θ that correspond to current class
            current_initial_theta = np.copy(self.theta[label_index].reshape(num_features, 1))
            # here we don't use label like 0, 1, 2..., we use 0 and 1,
            # thus we need convert label to 0 and 1
            current_labels = (self.labels == unique_label).astype(float)
            (current_theta, cost_history) = LogisticRegression.gradient_descent(self.data, current_labels, current_initial_theta, max_iterations)
            self.theta[label_index] = current_theta.T
            cost_histories.append(cost_history)

        return self.theta, cost_histories

    @staticmethod
    def gradient_descent(data, current_labels, current_initial_theta, max_iterations):
        cost_history = []
        num_features = data.shape[1]
        result = minimize(   #can refer to document
            # the object function to be optimized
            lambda current_theta: LogisticRegression.cost_function(data, current_labels, current_theta.reshape(num_features, 1)),
            # initial guess(weights)
            current_initial_theta,
            # optimizer
            method='CG',
            # method for computing the gradient vector
            jac=lambda current_theta: LogisticRegression.gradient_step(data, current_labels, current_theta.reshape(num_features, 1)),
            # save result
            callback=lambda current_theta: cost_history.append(LogisticRegression.cost_function(data, current_labels, current_initial_theta.reshape(num_features, 1))),
            options= {'maxiter': max_iterations}
        )
        if not result.success:
            raise ArithmeticError('Cannot minimize cost function' + result.message)
        optimized_theta = result.x.reshape(num_features, 1)
        return optimized_theta, cost_history

    @staticmethod
    def cost_function(data, labels, theta):
        num_examples = data.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        # all the samples with label 1(only have 0 and 1 since it's binary classification)
        y_is_set_cost = np.dot(labels[labels == 1].T, np.log(predictions[labels==1]))
        y_is_not_set_cost = np.dot(1 - labels[labels == 0].T, np.log(1 - predictions[labels==0]))
        # cross entropy
        cost = (-1/num_examples) * (y_is_not_set_cost + y_is_set_cost)

        return cost

    @staticmethod
    def hypothesis(data, theta):
        predictions = sigmoid(np.dot(data, theta))

        return predictions

    def gradient_step(data, labels, theta):
        num_examples = labels.shape[0]
        predictions = LogisticRegression.hypothesis(data, theta)
        label_diff = predictions - labels
        gradients = (1/num_examples) * np.dot(data.T, label_diff)

        return gradients.T.flatten()

    def predict(self, data):
        num_examples = data.shape[0]
        data_processed = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        prob = LogisticRegression.hypothesis(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, label in enumerate(self.unique_labels):
            class_prediction[max_prob_index==index] = label
        return class_prediction.reshape((num_examples, 1))

        
        
        
        
        
        
        


