# Linear Regression from Scratch

## What is Linear Regression?
Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It aims to find the best-fitting straight line (the regression line) that describes how the dependent variable changes as the independent variables change.

## Assumptions of Linear Regression
To ensure the validity of linear regression, the following assumptions should be met:
1. **Linearity**: The relationship between the independent and dependent variables should be linear.
2. **Independence**: Observations should be independent of each other.
3. **Homoscedasticity**: The residuals (the differences between the observed and predicted values) should have constant variance.
4. **Normality**: The residuals should be approximately normally distributed.
5. **No multicollinearity**: Independent variables should not be highly correlated with each other.

## Equation of Linear Regression
### Simple Linear Regression
The equation for a simple linear regression model with one independent variable is:

$$ y = \beta_0 + \beta_1 x + \epsilon $$

where:
- $$\ y \$$ is the dependent variable.
- $$\ x \$$ is the independent variable.
- $$\ \beta_0 \$$ is the y-intercept (the value of $$\ y \$$ when $$\ x \$$ is 0).
- $$\ \beta_1 \$$ is the slope of the regression line (the change in $$\ y \$$ for a one-unit change in $$\ x \$$).
- $$\ \epsilon \$$ is the error term (the difference between the observed and predicted values).

### Multiple Linear Regression
For multiple linear regression with multiple independent variables, the equation is:

$$\ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon \$$

## How to Calculate Linear Regression from Loss Function Using Gradient Descent

### 1. Define the Loss Function
The loss function measures how well the linear regression model fits the data. The most commonly used loss function is the Mean Squared Error (MSE), defined as:

$$\ \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 \$$

where:
- $$\ m \$$ is the number of observations.
- $$\ y_i \$$ is the actual value of the dependent variable for the $$\ i \$$-th observation.
- $$\ \hat{y}_i \$$ is the predicted value of the dependent variable for the $$\ i \$$-th observation.

### 2. Initialize Parameters
Initialize the parameters $$\ \beta_0 \$$ and $$\ \beta_1 \$$ (or $$\ \beta_j \$$ for multiple variables) with random values or zeros.

### 3. Calculate Gradients
Compute the gradients of the loss function with respect to the parameters. For the simple linear regression, the gradients are:

$$\ \frac{\partial \text{MSE}}{\partial \beta_0} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i) \$$

$$\ \frac{\partial \text{MSE}}{\partial \beta_1} = -\frac{2}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)x_i \$$

### 4. Update Parameters
Update the parameters using the gradients and a learning rate ($$\ \alpha \$$):

$$\ \beta_0 = \beta_0 - \alpha \frac{\partial \text{MSE}}{\partial \beta_0} \$$

$$\ \beta_1 = \beta_1 - \alpha \frac{\partial \text{MSE}}{\partial \beta_1} \$$

For multiple linear regression, the parameters are updated similarly for each $$\ \beta_j \$$.

### 5. Iterate Until Convergence
Repeat steps 3 and 4 until the parameters converge to values that minimize the loss function.
