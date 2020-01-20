

# -*- coding: utf-8 -*-
"""
File:   hw03.py
Author: Yixing Chen
Date:   18/10/2018
Desc:   
"""
import numpy as np
import matplotlib.pyplot as plt
import math 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance as sc 
import textwrap
from sklearn.decomposition import PCA
#generate data set that has features with widly varying range and variance.
N = 100
data = np.random.uniform((0,0),(3,10),(N,2))
X,y  = data[:,1], data[:,0]

plt.figure()
plt.scatter(X,y, color='red', marker = '^', alpha=0.5)
plt.show()
#compute the PCA transformed data set
scikit_pca = PCA(n_components = 2)
data_spca = scikit_pca.fit_transform(data)
plt.figure()
plt.scatter(data_spca[:,0], data_spca[:,1], color='blue', marker = 'o', alpha=0.5)
plt.show()
#compute the covariance matrices of original data set and data set after PCA
cov_data = np.cov(data.T)
cov_data_spca = np.cov(data_spca.T)
#whiten the data set
lam0=cov_data_spca[0,0]
lam1=cov_data_spca[1,1]
data_whitened = list(data_spca)
data_whitened = data_whitened@np.array([[1/np.sqrt(lam0),0],[0,1/np.sqrt(lam1)]],dtype=float)
#compute the covariance matrix of data set after whitening
cov_data_whi = np.cov(data_whitened.T)

plt.figure()
plt.scatter(data_whitened[:,0], data_whitened[:,1], color='green', marker = '.', alpha=0.5)
plt.show()
print('\nThe covariance matrix of original data set is: ', cov_data)
print('\nThe covariance matrix of data set after PCA is: ', cov_data_spca)
print('\nThe covariance matrix of data set after whitening is: ', cov_data_whi)