from regression import LinearRegression
from utils import mean_squared_error, r2_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create random regression data
X, y = make_regression(n_samples=1000, n_features=20, noise=100, random_state=28)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit & Predict Linear Regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predicted = regressor.predict(X_test)

# Calculate Regression Metrics
mse = mean_squared_error(y_predicted, y_test)
r2 = r2_error(y_predicted, y_test)

print("Mean Squared Error", mse)
print("R-squared Error", r2)
