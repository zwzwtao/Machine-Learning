import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

data = pd.read_csv('./data/world-happiness-report-2017.csv')
# 80% for training
train_data = data.sample(frac=0.8)
# drop training data, the remaining is for testing
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

# shouldn't use train_data[input_param_name], otherwise will cause error
# the returned value should be a 2D matrix
x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label='Training data')
plt.scatter(x_test, y_test, label='Test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('happiness_report')
plt.legend()
plt.show()

num_iteration = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
theta, losses = linear_regression.train(learning_rate, num_iteration)

print('initial loss:', losses[0])
print('loss after training:', losses[-1])

plt.plot(range(num_iteration), losses)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('GD')
plt.show()

pred_num = 100
x_pred = np.linspace(x_train.min(), x_train.max(), pred_num).reshape(pred_num, 1)
y_pred = linear_regression.predict(x_pred)

plt.scatter(x_train, y_train, label='Training data')
plt.scatter(x_test, y_test, label='Test data')
plt.plot(x_pred, y_pred, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('happiness_report')
plt.legend()
plt.show()
