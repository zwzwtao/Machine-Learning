import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

data = pd.read_csv('../data/iris.csv')
iris_types = ['SETOSA','VERSICOLOR','VIRGINICA']
x_axis = 'petal_length'
y_axis = 'petal_width'

for iris_type in iris_types:
    plt.scatter(
        data[x_axis][data['class']==iris_type],
        data[y_axis][data['class']==iris_type],
        label=iris_type
    )
plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
y_train = data['class'].values.reshape((num_examples, 1))

max_iterations = 1000
polynomial_degree = 0
sinusoid_degree = 0

logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree, sinusoid_degree)
theta, cost_histories = logistic_regression.train(max_iterations)
labels = logistic_regression.unique_labels

plt.plot(range(len(cost_histories[0])), cost_histories[0], labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], labels[1])
plt.plot(range(len(cost_histories[2])), cost_histories[2], labels[2])
plt.show()
















