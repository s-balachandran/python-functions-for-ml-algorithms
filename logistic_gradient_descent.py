# -*- coding: utf-8 -*-

import numpy as np

"""
Gradient Descent (with regularisation) for Logistic Regression

Note: Add column of ones as first column of X if 
not already done. (X = np.insert(X, 0, 1, axis = 1) )
This is for the intercept.
"""

#sigmoid function

def sigmoid(z):
    
    """
    Calculates sigmoid of a scalar or the elements of an array.
    
    Will be a value between 0 and 1.
    """
    g = 1/(1 + np.exp(-1*z))
    
    return g

#cost function for Logisitc Regression

def CostFunction(X, y, theta, reg):
    
    """
   Takes array of training examples' features, X, target variable, y, 
   array of weights, theta, and the regularisation parameter, reg.
    
    Calculates and returns cost of using particular theta.
    
    reg is the regularisation parameter. If regularisation
    not desired, set reg = 0.
    """
    #number of training examples
    m = len(y)
    
    #cost of using theta
    cost =  (-1/m)*(np.sum(y*np.log(sigmoid(X@theta)) + (1 - y)*np.log(1 - sigmoid(X@theta)))) 
    cost = cost + (reg/(2*m))*np.sum(theta[1:]**2)
    
    return cost

#Gradient Descent Function for specified precision
    
def GradientDescent(X, y, theta, reg, alpha, precision):
    
    """
    Takes X, y, theta, the regularisation parameter, reg,
    the learning rate, alpha, and the convergence precision 
    (difference between previous cost and current cost).
    
    Simulatenously updates theta by taking
    step down slope of cost function until precision reached.
    Size of step determined by learning rate, alpha.
    
    If regularisation not desired, set reg = 0.
    """
    
    m = len(y)
    cost_history = []
    
    previous_cost = CostFunction(X, y, theta, reg)
    cost_history.append(previous_cost)
    cost = 0
    i = 1
    
    while (previous_cost - cost) > precision:
        if i == 1:
            pass
        else:
            previous_cost = cost
        theta_temp = np.copy(theta)
        for j in range(len(theta)):
            if j == 0:
                theta[j] = theta[j] - (alpha/m)*np.sum((sigmoid(X@theta_temp) - y)*X[:,j])
            else:
                theta[j] = theta[j] - ((alpha/m)*np.sum((sigmoid(X@theta_temp) - y)*X[:,j]) + (reg/m)*theta[j])   
        cost = CostFunction(X,y,theta,reg)
        cost_history.append(cost)
        print(cost)
        i +=1
    
    return theta, cost_history


#Gradient Descent Function for specified number of iterations

def GradientDescentIter(X, y, theta, reg, alpha, num_iters):
    
    """
    Takes X, y, theta, the regularisation parameter, reg, 
    the learning rate, alpha, and the  number of iterations,
    num_iters.
    
    Simulatenously updates theta by taking
    step down slope of cost function for given num_iters.
    Size of step determined by learning rate, alpha.
    
    If regularisation not required, set reg = 0.
    """
    
    #number of training examples
    m = len(y)
    #list of cost for each iteration
    cost_history = []
    #for each iteration, simulateneously update parameters
    for iter in range(num_iters):
        theta_temp = np.copy(theta)
        for j in range(len(theta)):
            if j == 0:
                theta[j] = theta[j] - (alpha/m)*np.sum((sigmoid(X@theta_temp) - y)*X[:,j])
            else:
                theta[j] = theta[j] - ((alpha/m)*np.sum((sigmoid(X@theta_temp) - y)*X[:,j]) + (reg/m)*theta[j])
        cost = CostFunction(X, y, theta, reg)
        cost_history.append(cost)
        
    return theta, cost_history

#
