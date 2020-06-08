# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
K-Means Algorithm functions
"""

def findClosestCentroids(X, centroids):
    
    '''
    Calculates the closest centroid in centroids for each training example in X.
    Returns vector of centroid assignments for each training example.
    '''
    
    #set K
    K = centroids.shape[0]
    
    #vector of cluster assignments
    idx = np.zeros(X.shape[0])
    
    dist = np.zeros(K)
    for i in range(X.shape[0]):
        for k in range(K):
            dist[k] = np.sum((X[i,:] - centroids[k,:])**2)**0.5
        idx[i] = np.argmin(dist)
        
    return idx


def computeCentroids(X, idx, K):
    
    '''
    Calculates new centroids by computing the mean of the 
    data points assigned to each centroid. Returns matrix
    where each row is a new centroid's point.
    '''
    
    #no. of data points
    m = X.shape[0]
    #dimension of points
    n = X.shape[1]
    
    centroids = np.zeros((K,n))
    
    for k in range(K):
        count = 0
        s = np.zeros((1,n))
        for i in range(m):
            if idx[i] == k:
                s = s + X[i,:]
                count += 1
        centroids[k,:] = s/count
        
    return centroids


def RandInitialCentroids(X, K):
    
    '''
    Initializes K centroids by randomly selecting K points in X.
    '''
    
    centroids = np.zeros((K, X.shape[1]))
    
    #Randomly reorder the indicies of examples
    randidx = np.random.permutation(range(X.shape[0]))
    #Take the first K examples
    centroids = X[randidx[0:K],:]
    
    return centroids



def kMeansDistortion(X, idx, centroids):
    
    '''
    Calculates the average distance between the examples and the 
    centroid of the cluster to which each example has been assigned.
    '''
    
    #no. of data points
    m = X.shape[0]
    
    distortion = 0
    
    for i in range(X.shape[0]):
        closest = int(idx[i])
        distance = np.sum((X[i,:] - centroids[closest])**2)
        distortion = distortion + distance
        
    distortion = distortion/m
    
    return distortion


# k-Means algorithm function
    
def kMeans(X, K, max_iters):
           
    '''
    Run the kmeans algorithm for specified number of iterations 
    and returns final centroids, index of closest centroids for 
    each example (idx), final distortion, and distortion history.
    '''
    distortion_history = []
    distortion = 0
    centroids = RandInitialCentroids(X, K)       

    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        distortion = kMeansDistortion(X, idx, centroids)
        distortion_history.append(distortion)
        centroids = computeCentroids(X, idx, K)
        
    return centroids, idx, distortion, distortion_history   


#run k-means with specified different random initialisations and pick one with lowest distortion
def kMeansRuns(X, K, max_iters, init_runs):
    '''
    Run the kMeans algorithm for specified number of random initialisations, init_runs, 
    and return result with lowets distortion.
    '''
    for r in range(init_runs):
        if r == 0:
            centroids, index, distortion, distortion_hist = kMeans(X, K, max_iters)
            distortion_lowest = distortion
        else:
            current_centroids, current_index, distortion, current_distortion_hist = kMeans(X, K, max_iters)
            if distortion_lowest > distortion:
                centroids= current_centroids
                index = current_index
                distortion_lowest = distortion
                
    return centroids, index, distortion_lowest



       
       
       
       
       