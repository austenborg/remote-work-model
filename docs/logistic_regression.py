import numpy as np


def one_vs_all(X, y, num_classes, alpha=0.01, iters=1000, theta=None):
    '''
    Train a one vs. all logistic regression. Adapted from student work
    and sample solutions to Assignment 4.
    
    Inputs: 
        X               data matrix (2d array shape m x n)
        y               label vector with entries from 0 to num_classes - 1 (1d array length m)
        num_classes     number of classes (integer)
        theta           initial parameter vector (1d numpy array - length n)
        alpha           step size (scalar)
        iters           number of iterations (integer)

    Outputs:
        weight_vectors  matrix of weight vectors for each class 
                        weight vector for class c in the cth column (2d array shape n x num_classes)
        intercepts      vector of intercepts for all classes (1d array length num_classes)                           
    '''

    m, n = X.shape
    weight_vectors = np.zeros((n, num_classes))
    intercepts = np.zeros(num_classes)

    # calculating theta for each class
    for i in range(num_classes):
        y_class = (y == i)
        model, J_history = log_reg_gd(X, y_class, alpha, iters, theta=None)
        
        # the following three lines were taken from hw4.py solutions function train_one_vs_all()
        weight_vectors[:,i] = model.coef_.ravel()
        intercepts[i] = model.intercept_
        # print(weight_vectors[:,i])
        # note: the model functions were originally called on a model made like so:
            # model = linear_model.LogisticRegression(C=2./lambda_val, solver='lbfgs')
        # so if one_vs_all() isn't working, it might be because these don't work on our "homemade" log reg model

    return weight_vectors, intercepts


def predict_one_vs_all(X, weight_vectors, intercepts):
    '''
    Train a one vs. all logistic regression. Adapted from student work
    and sample solutions to Assignment 4.
    
    Inputs: 
        X                data matrix (2d array shape m x n)
        weight_vectors   matrix of weight vectors for each class 
                       sweight vector for class c in the cth column
                       (2d array shape n x num_classes)
        intercepts       vector of intercepts for all classes
                       (1d array length num_classes)   
                       
    Outputs:
        predictions      vector of predictions for examples in X
                       (1d array length m)            
    ''' 

    # matrix multiplication simultaneously makes predictions for all classes
    # vals = X.dot(weight_vectors) + intercepts;
    
    # vector of length m holding a class prediction for each data sample in X
    predictions = np.argmax(vals, axis=1)
    # argmax finds the index of largest value in an array, or in each row/column of an array
    
    return predictions 


def log_reg_gd(X, y, alpha, iters, theta=None):

    # X = (m x n), y = (n x 1)
    m,n = X.shape
    
    # initialize theta
    if theta is None:
        theta = np.zeros(n)

    # initialize cost function history
    J_history = np.zeros(iters)

    for i in range(0, iters):
        # calc gradient with the logistic function
        z = np.dot(X, theta)
        h = 1 / (1 + np.exp(-z))
        # compute gradient (vectorized) and update theta
        gradient = 2 * np.dot((h - y), X)
        theta = theta - (alpha * gradient)
    
        J_history[i] = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
    
    return theta, J_history
    