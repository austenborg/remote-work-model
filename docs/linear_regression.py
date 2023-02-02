import numpy as np

def linear_cost_function(X, y, theta):
    '''
    Compute the cost function for a particular data set and
    hypothesis (parameter vector). Adapted from student work
    and sample solutions to Assignment 2.

    Inputs:
        X       m x n data matrix
        y       training output (length m vector)
        theta   parameters (length n vector)
    Output:
        cost    the value of the cost function (scalar)
    '''

    err = np.dot(X, theta) - y
    cost = 0.5 * np.dot(err, err)
    return cost

def linear_gd( X, y, alpha, iters, theta=None ):
    '''
    Train a linear regression model by gradient descent.
    Adapted from student work and sample solutions to
    Assignment 2.

    Inputs:
        X       m x n data matrix
        y       training output (length m vector)
        alpha   step size
        iters   number of iterations
        theta   initial parameter values (length n vector; optional)

    Output:
        theta      learned parameters (length n vector)
        J_history  trace of cost function value in each iteration

    '''

    m, n = X.shape

    if theta is None:
        theta = np.zeros(n)

    # For recording cost function value during gradient descent
    J_history = np.zeros(iters)

    for i in range(0, iters):
        gradient = np.dot(X.T, np.dot(X, theta) - y)
        theta = theta - alpha * gradient

        # Record cost
        J_history[i] = linear_cost_function(X, y, theta)

    return theta, J_history
