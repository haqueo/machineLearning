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
from sklearn import cluster # delete this later
import matplotlib.pyplot as plt

def clean_data(unclean_data):
    dim = unclean_data.shape[2]
    n = unclean_data.shape[0]*unclean_data.shape[1]
    data_cleaned = np.reshape(unclean_data,(n,dim))
    return data_cleaned, dim, n

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
    
    X_cleaned, dim, n = clean_data(X)
    
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
                            max_iter=20,
                            verbose=1)
    clf.fit(X_cleaned)
    
    did_I_pass = np.all(np.isclose(clf.means_,new_params_mine["means"])) and np.all(np.isclose(clf.covariances_,new_params_mine["covariances"]))
    
    return did_I_pass


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

def test_k_means_clustering(X,K=10):
    
    X_cleaned, dim, n = clean_data(X)
    random.seed(1)
    centroids = X_cleaned[random.sample(range(n),K),:]

    
    
    clst = cluster.KMeans(n_clusters=K,init=centroids,max_iter=35,n_init=1, 
                          verbose=True)
    kmeans = clst.fit(X_cleaned)
    mycentroids, mydata_clustering = k_means_clustering(X,K)



if __name__ == "__main__":
    img = cv2.imread("/Users/Omar/Documents/Year4/machineLearning/coursework2/" + 
                 "data/question1/FluorescentCells.jpg")
    # parameters = run_GMM(img,K=4)
    centroids, data_clustering = k_means_clustering(img,K=4)
    myimg = centroids[data_clustering].reshape((1927,2560,3))/float(255)
    
    plt.imshow(myimg)
    cv2.imwrite('kmeans4.jpeg',myimg*255)
    