# -*- coding: utf-8 -*-

"""
Clustering Selection/Validation functions (Average Silhouette, Elbow Method)
"""
import numpy as np
import matplotlib.pyplot as plt

#import functions from file k-Means.py
from kMeans_algorithm import * 

"""
Average silhouette measures the quality of clustering.
We calculate the average silhouette for a range of number 
of clusters, and pick the the number of clusters which has
the highest average silhouette score.
"""

from scipy.spatial import distance

def SilhouetteScore(x_i, X, idx, K):
    '''
    For given training example (with index x_i), calculates  and returns its silhouette score.
    
    First, the function calculates the average distance, a_i, between the given training example
    and all other points in the cluster it belongs to. Then, the function calculates the average distance
    between the training example and all other points not in its own cluster, and picks the cluster with 
    the smallest average distance. Using a_i and b_i, it calculates the silhouette score of the given 
    training example.
    '''
    #calculate average distance between x_i and all points in its cluster
    
    #training example 
    point = X[x_i]
    #cluster index of training example
    idx_point = idx[x_i]
    
    #list of distances between point and other points in own cluster
    own_cluster_distances = np.empty(0)
    #loop over training examples' assigned cluster index, find points in 
    #own cluster and calculate euclidean distance
    for i in range(idx.shape[0]):
        if idx[i] == idx_point:
            own_cluster_distances = np.append(own_cluster_distances, distance.euclidean(point, X[i]))
    
    #average distance between point and all other points in own cluster
    a_i = np.sum(own_cluster_distances)/(own_cluster_distances.shape[0])
    
    #for each k in range K, calculate average distance between point and all other points in cluster k
    avg_cluster_distances = np.empty(0)
    #range of K without given trainig example's own cluster
    other_clusters = [r for r in range(K) if r != idx_point]
    
    for k in other_clusters:
        #distances between point and all points in cluster k
        k_distances = np.empty(0)
        #all points in cluster k
        k_cluster = X[idx==k]
        #number of points in cluster k
        k_len = k_cluster.shape[0]
        for n in range(k_len):
            k_distances = np.append(k_distances, distance.euclidean(point, k_cluster[n]))
        #average distance between point and all points in cluster k appended to
        #avg_cluster_distances array
        if k_len != 0:
            avg_cluster_distances = np.append(avg_cluster_distances, np.sum(k_distances)/k_len)
        else:
            avg_cluster_distances = np.append(avg_cluster_distances, 0)
        
        
    #find closest cluster in avg_cluster_distances
    b_i = np.min(avg_cluster_distances)
    
    silhouette_score = (b_i - a_i)/np.max([a_i, b_i])
    
    
    return silhouette_score 


def AverageSilhouette(X, idx,  K):
    '''
    Calculates and returns the average silhoutte for given number of clusters, K.
    
    Average silhouette is the average of the silhouette scores of all the training examples.
    '''
    silhouette_scores = np.empty(0)
    #loop over all training examples and calculate their silhouette score
    for i in range(X.shape[0]):
        silhouette_i = SilhouetteScore(i, X, idx, K)
        silhouette_scores = np.append(silhouette_scores, silhouette_i)
    
    #calculate average of all scores
    avg_silhouette = np.sum(silhouette_scores)/len(silhouette_scores)
    
    return avg_silhouette
        
"""
The following function runs k-Means for each K in the range 2 - K_range, 
calculates the average silhouette, and plots a graph of the average 
silhouette score for each K.
"""

def PlotAvgSilhouettes(X, K_range, max_iters, init_runs):
    
    '''
    Runs kMeans and plots the average silhouette scores for 
    each number of clusters in K_range.
    
    '''
    clusters_avg_sil = np.empty(0)
    #minimum of K_range must be 2
    for K in range(2, K_range+1):
        centroids, idx, distortion_lowest = kMeansRuns(X, K, max_iters, init_runs)
        k_avg_sil = AverageSilhouette(X, idx,  K)
        clusters_avg_sil = np.append(clusters_avg_sil, k_avg_sil)
        
    #plot graph of avg silhouettes of each cluster of size in range 2 - K_range
    plt.figure(figsize = (12.8, 9.6))
    plt.plot(np.arange(2,K_range+1,1), clusters_avg_sil)
    plt.title('Average Silhouette for number of clusters K')
    plt.xlabel('K')
    plt.ylabel('Average Silhouette')
    plt.show()
    
    return None


"""
The Elbow method can provide a quick snapshot of how clusters vary in terms of their distortion.
Sometimes useful for selecting number of clusters.
"""

def PlotElbow(X, K_range, max_iters, init_runs):
    
    '''
    Plots distortion for clusters of size in 2 - K_range.
    '''
    cluster_distortions = np.empty(0)
    for K in range(2, K_range+1):
       centroids, idx, distortion_lowest = kMeansRuns(X, K, max_iters, init_runs)
       cluster_distortions = np.append(cluster_distortions, distortion_lowest)
       
    #plot graph of distortion of each cluster of size in range 2 - K_range
    plt.figure(figsize = (12.8, 9.6))
    plt.plot(np.arange(2,K_range+1,1), cluster_distortions)
    plt.title('Distortion for number of clusters K')
    plt.xlabel('K')
    plt.ylabel('Distortion')
    plt.show()
    
    return None
