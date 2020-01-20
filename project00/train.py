# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:08:17 2018

@author: asus
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors

def train(filename):
    data_trainred = np.load(str(filename)+'.npy') #load the training data
    ground_truth = np.load('ground_truth.npy')

    gridnum=50 #Set numbers of pixels a grid contain horizontally to 10, 25, 50, 125...
    gridwidth=data_trainred.shape[0]//gridnum #calculate width of matrix network to 250
    network = np.zeros([gridwidth, gridwidth,5],dtype=float)
    red_coordinate = []
    cor_matrix=[]

    for i in range(data_trainred.shape[0]):
        for j in range(1,data_trainred.shape[1]):
            if data_trainred[i,j][0]>data_trainred[i,j][1]+30:  
                #select the pixels whose difference between R value and G value is greater than 30
                network[i//gridnum,j//gridnum,0]+=1
                #find the corresponding square of each pixels by selecting the quotient of coordinates divided by 25            
                red_coordinate.append([i,j])

    backup=list(red_coordinate)
    grid_red=[]
    #calculate the mean and variance of x coordinates and y coordinates of each grid
    for i in range(gridwidth):
        for j in range(gridwidth):
            for k in range(len(red_coordinate)):
                if red_coordinate[k][0]//gridnum==i and red_coordinate[k][1]//gridnum==j:
                    grid_red.append(red_coordinate[k])
                    backup.remove(red_coordinate[k])
                elif red_coordinate[k][0]//gridnum>i and len(grid_red)!=0:
                    network[i,j,1]=np.mean(grid_red,axis=0)[1] #x coordinates
                    network[i,j,2]=np.mean(grid_red,axis=0)[0]
                    network[i,j,3]=np.var(grid_red,axis=0)[1]  #y coordinates
                    network[i,j,4]=np.var(grid_red,axis=0)[0]
                    break
                
                
            grid_red=[]
            red_coordinate=list(backup)

            
    truth_number=[]
    x_mean=[]
    y_mean=[]
    x_var=[]
    y_var=[]
    #load the means of x coordinates, y coordinates, variances of x coordinates and y coordinates from the matrix
    for i in range(ground_truth.shape[0]):
        truth_number.append(network[ground_truth[i][1]//gridnum, ground_truth[i][0]//gridnum,0])
        x_mean.append(network[ground_truth[i][1]//gridnum, ground_truth[i][0]//gridnum,1])
        y_mean.append(network[ground_truth[i][1]//gridnum, ground_truth[i][0]//gridnum,2])
        x_var.append(network[ground_truth[i][1]//gridnum, ground_truth[i][0]//gridnum,3])
        y_var.append(network[ground_truth[i][1]//gridnum, ground_truth[i][0]//gridnum,4])
    
    
    #calculate the means of x coordinates, y coordinates, variances of x coordinates and y coordinates of ground truth
    mean_var=np.sum([x_var,y_var],axis=0)
    rate=truth_number/mean_var
    ratio=np.delete(rate,9)
    ratio_mean=np.mean(ratio)
    ratio_var=np.var(ratio)
    #return ratio_mean, ratio_var

#    plt.xlabel('sum variance')
#    plt.ylabel('red point numbers in a grid')
#
#
#    def fitdata(x,t,M):
#	'''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''	
#	#This needs to be filled in
#	   X = np.array([x**m for m in range(M+1)]).T
#	   w = np.linalg.inv(X.T@X)@X.T@t
#	   return w
#
#        
#    M = 1
#    w = fitdata(np.delete(mean_var,9),np.delete(truth_number,9),M)
#    xrange = np.arange(0,30,0.1)  #get equally spaced points in the xrange
#    X = np.array([xrange**m for m in range(w.size)]).T
#    esty = X@w #compute the predicted value
#    plt.figure(0)
#    plt.plot(xrange,esty)
#    plt.plot(np.delete(mean_var,9),np.delete(truth_number,9),'o')
#    plt.xlabel('Mean variance')
#    plt.ylabel('red point numbers in a grid')


    return ratio_mean, ratio_var

#for i in range(ground_truth.shape[0]):
  