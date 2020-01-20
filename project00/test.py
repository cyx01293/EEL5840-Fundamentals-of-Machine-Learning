# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:04:42 2018

@author: asus
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors

def test(filename,ratio_mean,ratio_var):
    data_test = np.load(str(filename)+'.npy')  #load the testing data

    gridnum=50  #Set numbers of pixels a grid contain horizontally to 10, 25, 50, 125...
    gridwidth=data_test.shape[0]//gridnum   #calculate width of matrix network to 250
    network = np.zeros([gridwidth, gridwidth,5],dtype=float)
    red_coordinate = []
    cor_matrix=[]

    for i in range(data_test.shape[0]):
        for j in range(1,data_test.shape[1]):
            if data_test[i,j][0]>data_test[i,j][1]+45:
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
               # temp=red_coordinate[k]
                    backup.remove(red_coordinate[k])
                elif red_coordinate[k][0]//gridnum>i and len(grid_red)!=0:
                    network[i,j,1]=np.mean(grid_red,axis=0)[1] #x coordinates
                    network[i,j,2]=np.mean(grid_red,axis=0)[0]
                    network[i,j,3]=np.var(grid_red,axis=0)[1]  #y coordinates
                    network[i,j,4]=np.var(grid_red,axis=0)[0]
                    break
                
                
            grid_red=[]
            red_coordinate=list(backup)
            
    test_result=[]
    test_result_coordinate=[]
    #estimate red cars' positions and draw a white pixels on the position coordinates of cars
    for i in range(gridwidth):
        for j in range(gridwidth):
            if (network[i,j,3]+network[i,j,4])!=0:
                ratio_cal=network[i,j,0]/(network[i,j,3]+network[i,j,4])
        #3u shuoyixia
                if np.abs(ratio_cal-np.sqrt(ratio_var))<2*ratio_mean and network[i,j,0]<120 and network[i,j,0]>10:
                    test_result.append([i,j])
                    test_result_coordinate.append([int(network[i,j,1]),int(network[i,j,2])])
                    data_test[int(network[i,j,2]),int(network[i,j,1])]=[255,255,255]
#            if ratio_cal<0.6 and ratio_cal>0.4 and network[i,j,0]<120 and network[i,j,0]>10:
#                if network[i,j,3]>network[i,j,4]:
#    test_result=[]
#    k=len(test_result)
#    for i in range(gridwidth):
#        for j in range(gridwidth):
#            if (network[i,j,3]+network[i,j,4])!=0:
#                ratio_cal=network[i,j,0]/(network[i,j,3]+network[i,j,4])
#        #3u shuoyixia
#                if np.abs(ratio_cal-np.sqrt(ratio_var))<2*ratio_mean and network[i,j,0]<120 and network[i,j,0]>10:
#                    test_result.append([i,j])
#                    k+=1
#                    if test_result[k-1][1]==j-1:
#                        test_result.remove[k]
#                        k-=1
#    n=len(test_result)
#    for j in range(n-1):
#        for i in range(n-1-j):
#            if test_result[i][1]>test_result[i+1][1]:
#                temp=test_result[i][1]
#                test_result[i][1]=test_result[i+1][1]
#                test_result[i+1][1]=temp
#    result_backup=list(test_result)
#    for i in range(n-1):
#        if test_result[i][1]!=test_result[i+1][1] or test_result[i][0]!=test_result[i+1][0]-1:
#            m=test_result[i][1]
#            n=test_result[i][0]
#            test_result_coordinate=[network[m,n,1],network[m,n,2]]
#            data_test[int(network[m,n,2]),int(network[m,n,1])]=[255,255,255]                    



        
 #   network[red_coordinate[i][0]//25,red_coordinate[i][1]//25,1]+=1
#    grid_red[red_coordinate[i][0]//25,red_coordinate[i][1]//25,0]=red_coordinate[i][0]
#    grid_red[red_coordinate[i][0]//25,red_coordinate[i][1]//25,1]=red_coordinate[i][1]
#    if 

    import imageio
    imageio.imwrite('test_sample_hou.png', data_test)
    return test_result_coordinate