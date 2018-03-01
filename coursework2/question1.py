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

def initialise_parameters(X,K=10):
    
    # initialise some key components
    random.seed(1)
    parameters = {}
    dim = X.shape[2]
    n = X.shape[0]*X.shape[1]
    X_cleaned = np.reshape(X,(n,dim))
    
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
    dim = X.shape[2]
    full_n = X.shape[0]*X.shape[1]
    X_cleaned = np.reshape(X,(full_n,dim))
    
    E = np.zeros((full_n, K))
    
    for n in range(full_n):
        for k in range(K):
            
            # this is the numerator, the denominator is just a scaling factor
            E[n,k] = params["mixtures"][k] * multivariate_normal.pdf(
                    x=X_cleaned[n,:], 
                    mean=params["means"][k], 
                    cov=params["covariances"][k])

    
    # normalise across rows
    
    E = E/E.sum(axis=1, keepdims=True)
    
    return E
    
    
    

def maximisation_step(X,expectations,K=10):
    pass




def run_GMM(X,K=10):
    
    params = initialise_parameters(X,K)
    
    for i in range(100):
        expectations = compute_expectations(X,K,params)
        params = maximisation_step(X,K,expectations)
    
    return params


    




if __name__ == "__main__":
    img = cv2.imread("/Users/Omar/Documents/Year4/machineLearning/coursework2/" + 
                 "data/question1/FluorescentCells.jpg")
    params = initialise_parameters(img,K=4)

    