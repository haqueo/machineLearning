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
import time # delete this later
from sklearn import mixture # delete this later

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
    dim = X.shape[2]
    full_n = X.shape[0]*X.shape[1]
    X_cleaned = np.reshape(X,(full_n,dim))
    
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



def test_npeinsum():
    
    Y = np.random.rand(100000,3)
    E = np.random.rand(100000,2)
    
    
    ## regular way
    t0 = time.time()
    sigma_1_regular = np.zeros((3,3))
    
    for n in range(100000):
        sigma_1_regular = sigma_1_regular + E[n,0] * np.outer(Y[n,:],Y[n,:])
    
    sigma_1_regular = sigma_1_regular / (np.sum(E[:,0]))
    t1 = time.time()
    
    
    ## now the einsum way
    t2 = time.time()
    sigma_1_einsum = np.einsum("a,ai,aj -> ij",E[:,0],Y,Y)
    t3 = time.time()
    
    
    ## now the dot product way
    t4 = time.time()
    sigma_1_dot = np.zeros((3,3))
    # covariance matrix
    for i in range(3):
        for j in range(3):
            sigma_1_dot[i,j] =  np.dot(E[:,0], 
                       np.multiply(Y[:,i] , 
                                  Y[:, j])) / (np.sum(E[:,0]))
    t5 = time.time()
    
    
    # now the matlab way
    sigma_1_matlab = np.zeros((3,3))
    Y = Y.T

    t6 = time.time()
    Y = Y * np.sqrt(E[:,0])    
    sigma_1_matlab = np.matmul(Y,Y.T) / np.sum(E[:,0])
    
    t7 = time.time()

    print("Regular way took" + str(t1-t0))
    print("Einsum way took" + str(t3-t2))
    print("Dot way took" + str(t5-t4))
    print("Matlab way took" + str(t7-t6))


    final_value = np.allclose(sigma_1_regular,sigma_1_dot)
    print(final_value)
    return final_value

def run_GMM(X,K=10,params = -1):
    
    if (params == -1):
        params = initialise_parameters(X,K)
    
    for i in range(20):
        expectations = compute_expectations(X,params,K)
        params = maximisation_step(X,expectations,K)
    
    return params


    
def test_run_GMM(X,K):
    
    dim = X.shape[2]
    full_n = X.shape[0]*X.shape[1]
    X_cleaned = np.reshape(X,(full_n,dim))
    
    initial_params = initialise_parameters(X,K)
    new_params_mine = run_GMM(img,K)
    precisions = np.zeros((K,3,3))
    for k in range(K):
        precisions[k] = np.linalg.inv(initial_params["covariances"][k])
    
    # turn initial params covariances into precisions
    
    clf = mixture.GaussianMixture(n_components=4, covariance_type='full',
                            weights_init = initial_params["mixtures"],
                            means_init = initial_params["means"],
                            precisions_init=precisions,
                            max_iter=20)
    clf.fit(X_cleaned)
    pass



if __name__ == "__main__":
    img = cv2.imread("/Users/Omar/Documents/Year4/machineLearning/coursework2/" + 
                 "data/question1/FluorescentCells.jpg")
    parameters = run_GMM(img,K=4)

    