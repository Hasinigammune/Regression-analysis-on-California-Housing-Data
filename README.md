# Regression analysis on California Housing Data

One of the most popular challenges in Kaggle is house price prediction. We have performed regression analysis on California Housing Dataset (https://www.kaggle.com/camnugent/california-housing-prices).

First we have implemented the Gradient Descent method in scratch.

# Gradient of the cost function for linear regression model
## Error Function

The function that we seek to minimize is the mean squared error, which is defined as the square of the difference between actual and predicted values. This is also referred to as the cost function. Let's call it $J$

$J = \displaystyle \frac{1}{2n} \sum_{i=1}^{n}(\text{Predicted Value} - \text{Actual Value})^2$

where the sum is over all $n$ data points.

Suppose the predicted model is $y = w_0 + w_1 x_1 + ... + w_p x_p$ where $w_0$ and $w_j$ are weights or coefficients for the bias and the variable $x_j$ and let actual value be $y_a$

$J = \displaystyle \frac{1}{2n} \sum_{i=1}^{n} [(w_0 + w_1 x_1 + ... w_p x_p) - y_a]^2$

To find the gradient w.r.t each of the w variables, we do the following:

$\displaystyle\frac{\partial J}{\partial w_0} = \displaystyle \frac{1}{n} \sum_{i=1}^{n}  [(w_0 + w_1 x_1 + ... w_p x_p) - y_a]$

$\displaystyle\frac{\partial J}{\partial w_j} = \displaystyle \frac{1}{n} \sum_{i=1}^{n} x_j [(w_0 + w_1 x_1 + ... w_p x_p) - y_a]$

We can use above gradient functions to minimize the convex error function.

The value $[(w_0 + w_1 x_1 + ... w_p x_p) - y_a]$ is called the residual i.e. difference between actual and predicted value for a data point.
