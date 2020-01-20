# -*- coding: utf-8 -*-
"""
File:   hw04.py
Author: 
Date:   
Desc:   
    
"""

import scipy.spatial.distance as sc 
import numpy as np
import matplotlib.pyplot as plt
import math 
import textwrap
import time

dataset2 = np.load('dataset3.npy')
subdataset1_0 = np.zeros((1,3))
subdataset1_1 = np.zeros((1,3))
subdataset1_2 = np.zeros((1,3))
for i in range(dataset2.shape[0]):
    if dataset2[i,2]==0:
       # subdataset1_0[i]=dataset1[i]
         subdataset1_0 = np.concatenate((subdataset1_0, dataset2[i,:].copy()[np.newaxis,:]),axis=0)
    elif dataset2[i,2]==1:
    #else:
         subdataset1_1 = np.concatenate((subdataset1_1, dataset2[i,:].copy()[np.newaxis,:]),axis=0)
#    else:
#         subdataset1_2 = np.concatenate((subdataset1_2, dataset2[i,:].copy()[np.newaxis,:]),axis=0)
subdataset1_0 = np.delete(subdataset1_0, 0, axis=0)
subdataset1_1 = np.delete(subdataset1_1, 0, axis=0)
subdataset1_2 = np.delete(subdataset1_2, 0, axis=0)


#x2=np.arange(1.4,1.6,0.01)
x2=np.arange(1.5,2,0.01)
x1=np.arange(-1.5,5,0.01)
#x3=30*x2-42
x3=-14*x2+26
#plot figure of relationship between root-mean-square and model order
plt.figure()
plt.plot(subdataset1_0[:,0],subdataset1_0[:,1],'yo')
plt.plot(subdataset1_1[:,0],subdataset1_1[:,1],'co')
#plt.plot(subdataset1_2[:,0],subdataset1_2[:,1],'mo')
plt.plot(x2,x3,'g-')
plt.plot(2.75*np.ones(x1.shape),x1,'g-')
plt.plot(3.5*np.ones(x1.shape),x1,'g-')
#plt.plot(x1,x3,'g-')
#plt.plot(3.7*np.ones(x1.shape),x1,'g-')
plt.xlabel('x1')
plt.ylabel('x2')
plt.savefig('1')

#pp1=plt.plot(n,Etrain,'go-')
plt.show

#boundaryset1_1=np.arange(-0.5,2.5,0.01)
#
##plot figure of relationship between root-mean-square and model order
#plt.figure()
#plt.plot(subdataset1_0[:,0],subdataset1_0[:,1],'bo')
#plt.plot(subdataset1_1[:,0],subdataset1_1[:,1],'ro')
#plt.plot(boundaryset1_1,0.5*np.ones(boundaryset1_1.shape),'g-')
#plt.plot(boundaryset1_1,1.5*np.ones(boundaryset1_1.shape),'g-')
#plt.xlabel('x1')
#plt.ylabel('x2')
#plt.savefig('2')
##pp1=plt.plot(n,Etrain,'go-')
#plt.show

plt.figure()
N1= np.arange(-1,2,0.01)
N2= -0.5*N1+0.5
N3= -0.5*N1+1.5
plt.plot(1,0,'mo')
plt.text(1,0,str((1,1)), family='serif', ha='right', wrap=True)
plt.plot(1,1,'co')
plt.text(1,1,str((0,1)), family='serif', ha='right', wrap=True)
plt.plot(0,0,'yo')
plt.text(0,0,str((-1,0)), family='serif', ha='right', wrap=True)
#plt.text(1,-0.8,str((1,0)), family='serif', ha='right', wrap=True)
plt.plot(N1,0.5*np.ones(N1.shape),'g-')
plt.plot(0.5*np.ones(N1.shape),N1,'g-')
plt.xlabel('N1')
plt.ylabel('N2')
plt.savefig('4')
#pp1=plt.plot(n,Etrain,'go-')
plt.show