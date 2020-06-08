# -*- coding: utf-8 -*-

"""
Gradient Descent Algorithm (with regularisation) for Linear Regression
"""

import numpy as np

"""
Note: Add column of ones as first column of X if 
not already done. (X = np.insert(X, 0, 1, axis = 1) )
This is for the intercept.


"""

#Cost Function

def CostFunction(X, y, theta, reg):
    
    """
    Takes array of training examples' features, X, target variable, y, 
    array of weights, theta, and the regularisation parameter, reg.
    
    Calculates and returns cost of using particular theta 
    (i.e. average sum of squared errors).
    
    reg is the regularisation parameter. If regularisation
    not desired, set reg = 0.
    
    """
    
    #number of training examples
    m = len(y)
    
    cost = (1/(2*m))*(np.sum(((X@theta) - y)**2))
    cost = cost + (reg/(2*m))*np.sum(theta[1:]**2)
    
    return cost


#Gradient Descent Function for specified number of iterations

def GradientDescentIter(X, y, theta, reg, alpha, num_iters):
    
    """
    Takes X, y, theta, the regularisation parameter, reg,
    the learning rate, alpha, and the number of iterations, 
    num_iters.
    
    Simulatenously updates theta by taking
    step down slope of cost function for given num_iters.
    Size of step determined by learning rate, alpha.
    
    reg is the regularisation parameter. If regularisation
    not desired, set reg = 0.
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
                theta[j] = theta[j] - (alpha/m)*np.sum((X@theta_temp - y)*X[:,j])
            else:
                theta[j] = theta[j] - ((alpha/m)*np.sum((X@theta_temp - y)*X[:,j]) + (reg/m)*theta[j]) 
        cost = CostFunction(X,y,theta)
        cost_history.append(cost)
        
    return theta, cost_history



#Gradient Descent Function for specified precision

def GradientDescent(X, y, theta, reg, alpha, precision):
  
    """
    Takes X, y, theta, the regularisation parameter, reg, 
    the learning rate, alpha, and the convergence precision 
    (difference between previous cost and current cost).
    
    Simulatenously updates theta by taking
    step down slope of cost function until precision reached.
    Size of step determined by learning rate, alpha.
    
    reg is the regularisation parameter. If regularisation
    not desired, set reg = 0.
    """
    #number of training examples
    m = len(y)
    #list of cost for each iteration
    cost_history = []
    
    previous_cost = CostFunction(X, y, theta)
    cost_history.append(previous_cost)
    cost = 0
    iter = 1
    
    #simultaneously  update parameter until desired convergence
    while (previous_cost - cost) > precision:
        if iter == 1:
            pass
        else:
            previous_cost = cost
        theta_temp = np.copy(theta)
        for j in range(len(theta)):
            theta[j] = theta[j] - ((alpha/m)*np.sum((X@theta_temp - y)*X[:,j]) + (reg/m)*theta[j]) 
        cost = CostFunction(X,y,theta)
        cost_history.append(cost)
        iter +=1
    
    return theta, cost_history

