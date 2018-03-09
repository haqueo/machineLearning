#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:39:34 2018

@author: Omar Haque

This document is written in Python 3
git repo: https://github.com/haqueo/machineLearning.git
please email me for access.
    
"""
# imports
import cv2
import random
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

## preliminary functions ##

def clean_data(unclean_data):
    dim = unclean_data.shape[2]
    n = unclean_data.shape[0]*unclean_data.shape[1]
    data_cleaned = np.reshape(unclean_data,(n,dim))
    return data_cleaned, dim, n


## Gaussian mixture models ##
def initialise_parameters(X,K=10):
    
    # initialise some key components
    random.seed(1)
    parameters = {}

    X_cleaned, dim, n = clean_data(X)
    
    # randomly split the dataset into K clusters.
    cluster_choices = np.array(random.choices(range(K),k=n))
    
    # initialise the arrays to be returned
    means = np.zeros((K,dim))
    covariances = np.zeros((K,dim,dim)) 
    mixtures = np.zeros(K)
    
    
    # go through this initial clustering and initialise the parameters
    for i in range(K):
        
        X_k = X_cleaned[cluster_choices == i,:]
        means[i,:] = np.mean(X_k,axis=0)
        covariances[i,:,:] = np.cov(X_k.transpose())
        mixtures[i] = X_k.shape[0]/n
    
    
    parameters["means"] = means
    parameters["covariances"] = covariances
    parameters["mixtures"] = mixtures
    
    
    return parameters

def compute_expectations(X,params,K=10):
    """
    :param X: The data matrix
    :param params: The current estimate for the parameters
    :param K: The number of mixtures
    """
    
    # clean the data into an appropriate shape
    X_cleaned, dim, full_n = clean_data(X)
    
    
    # initialise E, the matrix of expectations
    E = np.zeros((full_n, K))
    

    for k in range(K):
        # this is the numerator, the denominator is just a scaling factor
        E[:,k] = params["mixtures"][k] * multivariate_normal.pdf(
                x=X_cleaned, 
                mean=params["means"][k], 
                cov=params["covariances"][k])
    
    # normalise across rows
    E = E/E.sum(axis=1, keepdims=True)
    
    return E
    


def maximisation_step(X,expectations,K=10):

    # clean the data into an appropriate shape
    X_cleaned, dim, full_n = clean_data(X)
    
    # initialise params
    params = {}
    
    # initialise the arrays to be returned
    means = np.zeros((K,dim))
    covariances = np.zeros((K,dim,dim)) 
    mixtures = np.zeros(K)
    
    for k in range(K):
        # means
        means[k] = (X_cleaned.T * expectations[:,k]).sum(axis=1) / np.sum(expectations[:,k])
        # covariances
        Y = (X_cleaned - means[k]).T # scale the data by the mean and transpose
        Y = Y * np.sqrt(expectations[:,k])    # multiply by square root of responsibilities 
        covariances[k] = np.matmul(Y,Y.T) / np.sum(expectations[:,k]) # perform the calculation
        # mixtures
        mixtures[k] = np.sum(expectations[:,k]) / float(full_n)
                
    params["means"] = means
    params["covariances"] = covariances
    params["mixtures"] = mixtures
    
    return params


def run_GMM(X,K=10,params = -1):
    
    if (params == -1):
        params = initialise_parameters(X,K)
    
    for i in range(20):
        expectations = compute_expectations(X,params,K)
        params = maximisation_step(X,expectations,K)
    
    return params

## K means ##


def get_closest_cluster(data, centroids,K):
    """ 
    Given the data and current centroids estimate, calculate the cluster for 
    each datapoint in data
    Return data clustering
    """
    
    norms_matrix = np.zeros((data.shape[0],K))
    for k in range(K):
        norms_matrix[:,k] = np.linalg.norm(data - centroids[k],axis=1)
    data_clustering = np.argmin(norms_matrix,axis=1)        

    return data_clustering

def update_centroids(data, data_clustering,K):
    """ 
    Given the data and current data clusters, recompute the centroids for 
    each cluster
    Return updated centroids
    """
    
    updated_centroids = np.zeros((K,data.shape[1]))
    
    for k in range(K):
        kth_points = data[data_clustering == k,:]
        centroid_kth = np.mean(kth_points,axis=0)
        updated_centroids[k] = centroid_kth
        
    return(updated_centroids)

    

def k_means_clustering(X,K=10,maxiter=200):

    # clean the data into an appropriate shape
    X_cleaned, dim, n = clean_data(X)
    
    ## initialise centroids
    random.seed(1)
    centroids = X_cleaned[random.sample(range(n),K),:]
    # initialise convergence criteria
    converged = False
    
    for i in range(maxiter):
        
        #bookkeeping
        old_centroids = centroids
        
        # data assignment step
        data_clustering = get_closest_cluster(X_cleaned, centroids,K)
        # centroids update step
        centroids = update_centroids(X_cleaned, data_clustering,K)
        
        if (np.allclose(centroids,old_centroids,rtol=1e-3)):
            print("converged at iteration " + str((i+1)))
            converged = True
            break
        
    
    if (not converged):
        print("Warning, clusters did not converge.")
    
    return centroids, data_clustering

## main ##

if __name__ == "__main__":
    img = cv2.imread("/Users/Omar/Documents/Year4/machineLearning/coursework2/" + 
                 "data/question1/FluorescentCells.jpg")
    parameters = run_GMM(img,K=4)
    centroids, data_clustering = k_means_clustering(img,K=4)
    myimg_kmeans = centroids[data_clustering].reshape((1927,2560,3))/float(255)
    
    