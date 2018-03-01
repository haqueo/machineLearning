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
    
    
    # initialise E, the matrix of expectations
    E = np.zeros((full_n, K))
    

    for k in range(K):
        # this is the numerator, the denominator is just a scaling factor
        E[:,k] = params["mixtures"][k] * multivariate_normal.pdf(
                x=X_cleaned, 
                mean=params["means"][k], 
                cov=params["covariances"][k])

    # thought: maybe I should be working in log scale to prevent underflow
    
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
        means[k,:] = (X_cleaned.T * expectations[:,k]).sum(axis=1) / np.sum(expectations[:,k])
#        covariances[k,:,:] = 
#
#    for k in range(K):
#
#        means[k,:] = (np.array([[float(np.dot(X_cleaned[:, 0], E[:, k]))],
#                                [float(np.dot(X_cleaned[:, 1], E[:, k]))],
#                                [float(np.dot(X_cleaned[:, 2], E[:, k]))]])).reshape(dim)        
#            
#        params
#        # means #
#        A['means_' + str(k)] = (np.array([[float(np.dot(B[0, :], E[:, (k - 1)]))],
#                                          [float(np.dot(B[1, :], E[:, (k - 1)]))]]) / float(
#            np.sum(E[:, (k - 1)]))).reshape(2)
#    
#        # covars #
#        A['covars_' + str(k)] = np.array(
#            [[np.dot(E[:, (k - 1)], np.multiply(B[0, :] - A['means_' + str(k)][0], B[0, :] - A['means_' + str(k)][0])),
#              np.dot(E[:, (k - 1)], np.multiply(B[0, :] - A['means_' + str(k)][0], B[1, :] - A['means_' + str(k)][1]))],
#             [np.dot(E[:, (k - 1)], np.multiply(B[1, :] - A['means_' + str(k)][1], B[0, :] - A['means_' + str(k)][0])),
#              np.dot(E[:, (k - 1)],
#                     np.multiply(B[1, :] - A['means_' + str(k)][1], B[1, :] - A['means_' + str(k)][1]))]]) / float(
#            np.sum(E[:, (k - 1)]))
#    
#        # mixCoeff #
#        A['mixCoeff_' + str(k)] = np.sum(E[:, (k - 1)]) / float(full_n)
#    
    
    return params



def test_npeinsum():
    
    Y = np.reshape(range(3*100000),(100000,3))
    E = np.random.rand(100000,2)
    
    
    ## regular way
    t0 = time.time()
    sigma_1_regular = np.zeros((3,3))
    
    for n in range(4):
        sigma_1_regular = sigma_1_regular + E[n,0] * np.outer(Y[n,:],Y[n,:])
    t1 = time.time()
    
    
    ## now the einsum way
    t2 = time.time()
    sigma_1_einsum = np.einsum("a,ai,aj -> ij",E[:,0],Y,Y)
    t3 = time.time()

    print("Regular way took" + str(t1-t0))
    print("Einsum way took" + str(t3-t2))


    final_value = np.allclose(sigma_1_regular,sigma_1_einsum)
    print(final_value)
    return final_value

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
    
    E = compute_expectations(img,params,K=4)

    