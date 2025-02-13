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
    return data_cleaned


## Gaussian mixture models ##
def initialise_parameters(X_cleaned,K=10):
    
    # initialise some key components
    random.seed(1)
    parameters = {}
    dim = X_cleaned.shape[1]
    n = X_cleaned.shape[0]
    
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

def compute_expectations(X_cleaned,params,K=10):
    """
    :param X: The data matrix
    :param params: The current estimate for the parameters
    :param K: The number of mixtures
    """
    

    full_n = X_cleaned.shape[0]
    
    
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
    


def maximisation_step(X_cleaned,expectations,K=10):

    # clean the data into an appropriate shape
    dim = X_cleaned.shape[1]
    full_n = X_cleaned.shape[0]
    
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


def run_GMM(X_cleaned,K=10,params = -1,seed=1,maxiter=40):
    
    if (params == -1):
        random.seed(seed)
        params = initialise_parameters(X_cleaned,K)
    
    for i in range(maxiter):
        
        old_params = params
        
        expectations = compute_expectations(X_cleaned,params,K)
        params = maximisation_step(X_cleaned,expectations,K)
        
        if (np.allclose(params["means"],old_params["means"],rtol=1e-3)):
            print("converged at iteration " + str((i+1)))
            break
    
    return params, expectations

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

    
def log_likelihood(cleaned_data,K,parameters,expectations):
    """ 
    Compute the log likelihood for a Gaussian Mixture Model
    """
    pdfs = multivariate_normal.pdf(
                x=cleaned_data, 
                mean=parameters["means"][0], 
                cov=parameters["covariances"][0]) * parameters["mixtures"][0]
    
    
    return np.sum(np.log(np.divide(pdfs,expectations[:,0])))
    
    
def count_cells(cleaned_dataset, cell_lower_bound, cell_upper_bound):
    """
    We use the AIC methodology described here: 
        https://uk.mathworks.com/help/stats/tune-gaussian-mixture-models.html
    in order to find the number of cells, fine tuning the number of components
    in a gaussian mixture model. Can initialise with k means.
    """
    
    n = cleaned_dataset.shape[0]
    AIC_list = {}
    BIC_list = {}
    parameters = {}
    
    
    # initialise using kmeans
    centroids, data_clustering = k_means_clustering(cleaned_dataset,K=cell_lower_bound)
    # initialise the arrays to be returned
    means = centroids
    covariances = np.zeros((cell_lower_bound,2,2)) 
    mixtures = np.zeros(cell_lower_bound)

    
    # go through this initial clustering and initialise the parameters
    for i in range(cell_lower_bound):
        X_k = cleaned_dataset[data_clustering == i,:]
        covariances[i,:,:] = np.cov(X_k.transpose())
        mixtures[i] = X_k.shape[0]/n
    
    
    parameters["means"] = means
    parameters["covariances"] = covariances
    parameters["mixtures"] = mixtures
    
    
    for k_num in range(cell_lower_bound,cell_upper_bound+1):
        print("k is " + str(k_num))

        params, expectations = run_GMM(cleaned_dataset,K=k_num,params=parameters)
        log_lik = log_likelihood(cleaned_dataset,k_num,params,expectations)
        # compute AIC
        AIC_list[k_num] = -2*log_lik + 2 * k_num
        # compute BIC
        BIC_list[k_num] = -2*log_lik + k_num * np.log(n)
        
        parameters = params
        
    return AIC_list, BIC_list, params, expectations
    
    
    

def k_means_clustering(X_cleaned,K=10,maxiter=100):

    # get dimensions of data
    n = X_cleaned.shape[0]
    dim = X_cleaned.shape[1]
    
    ## initialise centroids randomly
    random.seed(1)
    centroids = X_cleaned[random.sample(range(n),K),:]
    
    # initialise convergence criteria
    converged = False
    
    for i in range(maxiter):
        
        # bookkeeping, simply keep track of the old centroids
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


def produce_images():
    img = cv2.imread("/Users/Omar/Documents/Year4/machineLearning/coursework2/" + 
             "data/question1/FluorescentCells.jpg")
    img_cleaned = clean_data(img)
    ## k means
    k_values = [5,3,2]
    kmeans_images = np.zeros((len(k_values),1927,2560,3))
    gmm_images = np.zeros((len(k_values),1927,2560,3))
    
    for i, k in enumerate(k_values):
        centroids, data_clustering = k_means_clustering(img_cleaned,K=k)
        kmeans_images[i] = centroids[data_clustering].reshape((1927,2560,3))/float(255)
        plt.imsave("kmeans_" + str(k),kmeans_images[i])
        parameters, expectations = run_GMM(img_cleaned,K=k,maxiter=40)
        gmm_images[i] = parameters["means"][np.argmax(expectations,axis=1)].reshape(1927,2560,3)/float(255)
        plt.imsave("gmm_" + str(k),gmm_images[i])
        





## main ##

if __name__ == "__main__":
    img = cv2.imread("/Users/Omar/Documents/Year4/machineLearning/coursework2/" + 
                 "data/question1/FluorescentCells.jpg")
    img_cleaned = clean_data(img)
    
    parameters, expectations = run_GMM(img_cleaned,K=3)
    myimg_gmm = parameters["means"][np.argmax(expectations,axis=1)]
    
    cell_colour = np.array([ 155.39698821,  175.87112501,   93.20177927])    
    myimg_gmm_blackwhite = np.where(np.isclose(myimg_gmm,cell_colour).all(axis=1),1,0).reshape((1927,2560))
#    
    cleaned_dataset = np.column_stack(((np.where(myimg_gmm_blackwhite == 1))[0],np.where(myimg_gmm_blackwhite == 1)[1]))    
#    
    AICS, BICS, params, expectations = count_cells(cleaned_dataset,82,90)
#    
#    plt.scatter(cleaned_dataset[:,0],cleaned_dataset[:,1],s=0.1)
#    plt.scatter(centroids[:,0],centroids[:,1],color="red")
#    
#    
#    
#    ### ERRORS:
#    ## I've chosen all the NON cells, not the actual cells
#    ## I think column stack doesn't work properly
#    
#
##    
##    
##    
##    myimg_gmm = parameters["means"][np.argmax(expectations,axis=1)].reshape((1927,2560,3))
##    cv2.imwrite('kmeans4test.jpeg',myimg_kmeans*255)
##    
#
##    cv2.imwrite('gmm32.blackwhite.jpg',myimg_gmm_blackwhite*255)
##    
#
##    
##    
##
