# Name: Kanishka S
# Roll No: 2023115114
# File: id3
# Date: 09-09-2025

import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0


def perceptron_train(X, y, lr=0.1, epochs=10):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        for i in range(n_samples):
            linear_output = np.dot(X[i], weights) + bias
            y_pred = step_function(linear_output)
            error = y[i] - y_pred

            weights += lr * error * X[i]
            bias += lr * error

    return weights, bias


def perceptron_predict(X, weights, bias):
    y_pred = []
    for x in X:
        linear_output = np.dot(x, weights) + bias
        y_pred.append(step_function(linear_output))
    return np.array(y_pred)


# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

weights, bias = perceptron_train(X, y)
predictions = perceptron_predict(X, weights, bias)

print("Weights:", weights)
print("Bias:", bias)
print("Predictions:", predictions)
