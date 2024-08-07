#Steps
# y = wx +  b
# Initialize w = 0
# Initialize b = 0
# Predict result using y = wx + b
# Calculate error (MSE)
# Use Gradient Descent to figure out the new w and b
# Repeat iters times

import numpy as np

class LinearRegression:
    def __init__(self, alpha=0.1, iters=1000) -> None:
        # Initialize class varibles
        self.alpha = alpha
        self.iters = iters
        self.weights = None
        self.bias = None

    def fit(self, X, y) -> None:
        n_samples, n_features = X.shape

        # Initialize weights and biases as 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Execute iters times
        for _ in range(self.iters):
            # y = wx + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Gradient Descent for MSE
            dw = (1/n_samples) * 2 * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * 2 * np.sum(y_pred - y)

            # Update weights and biases
            self.weights = self.weights - self.alpha * dw
            self.bias = self.bias - self.alpha * db

    def predict(self, X):
        # y = wx + b
        return np.dot(X, self.weights) + self.bias 