# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:13:48 2018

@author: asus
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import textwrap
import time

#network2 for dataset1
filename='dataset1'
weights1=np.array([[0,1],[0,1]])    #set the weights and bias of the hand-designed network
bias1=np.array([[-1.5],[-0.5]])
weights2=np.array([1,-1])
bias2=0.5
dataset = np.load(str(filename)+'.npy')
dataset = np.delete(dataset,2,axis=1)   #remove the label of the dataset
nPts = dataset.shape[0]

N1 = (dataset@weights1[0,:]+bias1[0])
N2 = (dataset@weights1[1,:]+bias1[1])
for i in range(N1.shape[0]):           #activate the neural network function
    if N1[i]>=0:
        N1[i]=1
    else:
        N1[i]=-1
    if N2[i]>=0:
        N2[i]=1
    else:
        N2[i]=-1

y_dataset1_NN1 = (N1*weights2[0]+N2*weights2[1]+bias2) #calculate the output of neural network
for i in range(y_dataset1_NN1.shape[0]):
    if y_dataset1_NN1[i]>=0:
        y_dataset1_NN1[i]=1
    else:
        y_dataset1_NN1[i]=-1
plt.figure()
for i in range(dataset.shape[0]):              #generate the plot with the output value of the network
    if y_dataset1_NN1[i]==1:
        plt.plot(dataset[i,0],dataset[i,1],'ro')
    else:
        plt.plot(dataset[i,0],dataset[i,1],'bo')
plt.xlabel('x1')
plt.ylabel('x2')

x1=np.arange(-0.5,2.5,0.01)                  #plot the boundary line
plt.plot(x1,0.5*np.ones(x1.shape),'g-')
plt.plot(x1,1.5*np.ones(x1.shape),'g-')
plt.text(0.3,1.6,str('N1'), family='serif', ha='right', wrap=True)
plt.text(0.3,0.4,str('N2'), family='serif', ha='right', wrap=True)
plt.savefig('01')

#Network2 for dataset1
filename='dataset1'
weights1=np.array([[1,0],[1,0]])       #set the weights and bias of the hand-designed network
bias1=np.array([[-0.5],[-1.5]])
weights2=np.array([1,-1])
bias2=-0.5
dataset = np.load(str(filename)+'.npy')
dataset = np.delete(dataset,2,axis=1)   #remove the label of the dataset
nPts = dataset.shape[0]

N1 = (dataset@weights1[0,:]+bias1[0])
N2 = (dataset@weights1[1,:]+bias1[1])
for i in range(N1.shape[0]):           #activate the neural network function
    if N1[i]>=0:
        N1[i]=1
    else:
        N1[i]=-1
    if N2[i]>=0:
        N2[i]=1
    else:
        N2[i]=-1

y_dataset1_NN2 = (N1*weights2[0]+N2*weights2[1]+bias2) #calculate the output of neural network
for i in range(y_dataset1_NN1.shape[0]):
    if y_dataset1_NN2[i]>=0:
        y_dataset1_NN2[i]=1
    else:
        y_dataset1_NN2[i]=-1
plt.figure()
for i in range(dataset.shape[0]):          #generate the plot with the output value of the network
    if y_dataset1_NN2[i]==1:
        plt.plot(dataset[i,0],dataset[i,1],'ro')
    else:
        plt.plot(dataset[i,0],dataset[i,1],'bo')
plt.xlabel('x1')
plt.ylabel('x2')

x1=np.arange(-0.5,2.5,0.01)             #plot the boundary line
plt.plot(0.5*np.ones(x1.shape),x1,'g-')
plt.plot(1.5*np.ones(x1.shape),x1,'g-')
plt.text(0.3,2,str('N1'), family='serif', ha='right', wrap=True)
plt.text(1.8,2,str('N2'), family='serif', ha='right', wrap=True)
plt.savefig('02')

#Network1 for dataset2
filename='dataset2'
weights1=np.array([[1,1],[0,1]])           #set the weights and bias of the hand-designed network
bias1=np.array([[0],[-0.5]])
weights2=np.array([[1,0],[0,1]])
bias2=np.array([[-0.5],[-0.5]])
dataset = np.load(str(filename)+'.npy')
dataset = np.delete(dataset,2,axis=1)   #remove the label of the dataset
nPts = dataset.shape[0]

N1 = (dataset@weights1[0,:]+bias1[0])
N2 = (dataset@weights1[1,:]+bias1[1])
for i in range(N1.shape[0]):           #activate the neural network function
    if N1[i]<0:
        N1[i]=0
    else:
        N1[i]=1
    if N2[i]<0:
        N2[i]=0
    else:
        N2[i]=1

y1_dataset2_NN1 = (N1*weights2[0,0]+N2*weights2[0,1]+bias2[0]) #calculate the output y1 of neural network
y2_dataset2_NN1 = (N1*weights2[1,0]+N2*weights2[1,1]+bias2[1]) #calculate the output y2 of neural network
for i in range(y1_dataset2_NN1.shape[0]):             #activate the neural network function
    if y1_dataset2_NN1[i]<0:
        y1_dataset2_NN1[i]=0
    else:
        y1_dataset2_NN1[i]=1
    if y2_dataset2_NN1[i]<0:
        y2_dataset2_NN1[i]=0
    else:
        y1_dataset2_NN1[i]=1
plt.figure()
for i in range(dataset.shape[0]):              #generate the plot with the output value of the network
    if y1_dataset2_NN1[i]==0:
        plt.plot(dataset[i,0],dataset[i,1],'yo')
    elif y2_dataset2_NN1[i]>0:
        plt.plot(dataset[i,0],dataset[i,1],'co')
    else:
        plt.plot(dataset[i,0],dataset[i,1],'mo')
plt.xlabel('x1')
plt.ylabel('x2')

x1=np.arange(-1.5,1.5,0.01)                 #plot the boundary line
x2=-x1
plt.plot(0.5*np.ones(x1.shape),x1,'g-')
plt.plot(x1,x2,'g-')
plt.text(-0.5,1,str('N1'), family='serif', ha='right', wrap=True)
plt.text(0.7,1,str('N2'), family='serif', ha='right', wrap=True)
plt.savefig('03')

#Network2 for dataset2
filename='dataset2'
weights1=np.array([[1,0],[0,1]])      #set the weights and bias of the hand-designed network
bias1=np.array([[-0.5],[-0.5]])
weights2=np.array([[1,0],[0.5,1]])
bias2=np.array([[-0.5],[-0.5]])
dataset = np.load(str(filename)+'.npy')
dataset = np.delete(dataset,2,axis=1)   #remove the label of the dataset
nPts = dataset.shape[0]

N1 = (dataset@weights1[0,:]+bias1[0])
N2 = (dataset@weights1[1,:]+bias1[1])
for i in range(N1.shape[0]):           #activate the neural network function
    if N1[i]<0:
        N1[i]=0
    else:
        N1[i]=1
    if N2[i]<0:
        N2[i]=0
    else:
        N2[i]=1

y1_dataset2_NN2 = (N1*weights2[0,0]+N2*weights2[0,1]+bias2[0]) #calculate the output y1 of neural network
y2_dataset2_NN2 = (N1*weights2[1,0]+N2*weights2[1,1]+bias2[1]) #calculate the output y2 of neural network
for i in range(y1_dataset2_NN2.shape[0]):             #activate the neural network function
    if y1_dataset2_NN2[i]<0:
        y1_dataset2_NN2[i]=0
    else:
        y1_dataset2_NN2[i]=1
    if y2_dataset2_NN2[i]<0:
        y2_dataset2_NN2[i]=0
    else:
        y2_dataset2_NN2[i]=1
plt.figure()
for i in range(dataset.shape[0]):            #generate the plot with the output value of the network
    if y1_dataset2_NN2[i]>0:
        plt.plot(dataset[i,0],dataset[i,1],'mo')
    elif y2_dataset2_NN2[i]==0:
        plt.plot(dataset[i,0],dataset[i,1],'yo')
    else:
        plt.plot(dataset[i,0],dataset[i,1],'co')
plt.xlabel('x1')
plt.ylabel('x2')

x1=np.arange(-0.5,1.5,0.01)               #plot the boundary line
x2=np.arange(-1.5,1.5,0.01)
plt.plot(0.5*np.ones(x1.shape),x1,'g-')
plt.plot(x2,0.5*np.ones(x2.shape),'g-')
plt.text(-0.5,0.6,str('N1'), family='serif', ha='right', wrap=True)
plt.text(0.7,1,str('N2'), family='serif', ha='right', wrap=True)
plt.savefig('04')


#Network1 for dataset3
filename='dataset3'
weights1=np.array([[1,0],[1,0],[1,0]])        #set the weights and bias of the hand-designed network
bias1=np.array([[-1.5],[-2.75],[-3.5]])
weights2=np.array([1,-0.5,1])
bias2=0
dataset = np.load(str(filename)+'.npy')
dataset = np.delete(dataset,2,axis=1)   #remove the label of the dataset
nPts = dataset.shape[0]

N1 = (dataset@weights1[0,:]+bias1[0])
N2 = (dataset@weights1[1,:]+bias1[1])
N3 = (dataset@weights1[2,:]+bias1[2])
for i in range(N1.shape[0]):           #activate the neural network function
    if N1[i]<0:
        N1[i]=-1
    else:
        N1[i]=1
    if N2[i]<0:
        N2[i]=-1
    else:
        N2[i]=1
    if N3[i]<0:
        N3[i]=-1
    else:
        N3[i]=1

y_dataset3_NN1 = (N1*weights2[0]+N2*weights2[1]+N3*weights2[2]+bias2) #calculate the output y1 of neural network

for i in range(y_dataset3_NN1.shape[0]):             #activate the neural network function
    if y_dataset3_NN1[i]<0:
        y_dataset3_NN1[i]=-1
    else:
        y_dataset3_NN1[i]=1
plt.figure()
for i in range(dataset.shape[0]):                 #generate the plot with the output value of the network
    if y_dataset3_NN1[i]>0:
        plt.plot(dataset[i,0],dataset[i,1],'co')
    else:
        plt.plot(dataset[i,0],dataset[i,1],'yo')
plt.xlabel('x1')
plt.ylabel('x2')

x1=np.arange(-0.5,5,0.01)                 #plot the boundary line
x2=np.arange(-1.5,1.5,0.01)
plt.plot(1.5*np.ones(x1.shape),x1,'g-')
plt.plot(2.75*np.ones(x1.shape),x1,'g-')
plt.plot(3.5*np.ones(x1.shape),x1,'g-')
plt.text(1.6,4.8,str('N1'), family='serif', ha='right', wrap=True)
plt.text(2.6,4.8,str('N2'), family='serif', ha='right', wrap=True)
plt.text(3.6,4.8,str('N3'), family='serif', ha='right', wrap=True)
plt.savefig('05')


#Network2 for dataset3
filename='dataset3'
weights1=np.array([[14,1],[1,0],[1,0]])     #set the weights and bias of the hand-designed network
bias1=np.array([[-26],[-2.75],[-3.5]])
weights2=np.array([1,-1,1])
bias2=0
dataset = np.load(str(filename)+'.npy')
dataset = np.delete(dataset,2,axis=1)   #remove the label of the dataset
nPts = dataset.shape[0]

N1 = (dataset@weights1[0,:]+bias1[0])
N2 = (dataset@weights1[1,:]+bias1[1])
N3 = (dataset@weights1[2,:]+bias1[2])
for i in range(N1.shape[0]):           #activate the neural network function
    if N1[i]<0:
        N1[i]=-1
    else:
        N1[i]=1
    if N2[i]<0:
        N2[i]=-1
    else:
        N2[i]=1
    if N3[i]<0:
        N3[i]=-1
    else:
        N3[i]=1

y_dataset3_NN2 = (N1*weights2[0]+N2*weights2[1]+N3*weights2[2]+bias2) #calculate the output y1 of neural network

for i in range(y_dataset3_NN1.shape[0]):             #activate the neural network function
    if y_dataset3_NN2[i]<0:
        y_dataset3_NN2[i]=-1
    else:
        y_dataset3_NN2[i]=1
plt.figure()
for i in range(dataset.shape[0]):               #generate the plot with the output value of the network
    if y_dataset3_NN2[i]>0:
        plt.plot(dataset[i,0],dataset[i,1],'co')
    else:
        plt.plot(dataset[i,0],dataset[i,1],'yo')
plt.xlabel('x1')
plt.ylabel('x2')

x1=np.arange(-0.5,5,0.01)
x2=np.arange(1.5,2,0.01)
x3=-14*x2+26
plt.plot(x2,x3,'g-')                       #plot the boundary line
plt.plot(2.75*np.ones(x1.shape),x1,'g-')
plt.plot(3.5*np.ones(x1.shape),x1,'g-')
plt.text(1.6,4.8,str('N1'), family='serif', ha='right', wrap=True)
plt.text(2.6,4.8,str('N2'), family='serif', ha='right', wrap=True)
plt.text(3.6,4.8,str('N3'), family='serif', ha='right', wrap=True)
plt.savefig('06')