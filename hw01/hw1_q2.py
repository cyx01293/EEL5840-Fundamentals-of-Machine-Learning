# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:38:33 2018

@author: asus
"""

"""
===============================================================================
===============================================================================
============================ Question 2 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Variable Declaration ========================== """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math


plt.close('all') #close any open plots


def plotprior(sigma,mu):
    '''plotBeta(a=1,b=1): Plot plot beta distribution with parameters a and b'''
    xrange = np.arange(0,1,0.001)  #get equally spaced points in the xrange
    prior = 1/np.sqrt(2*np.pi*sigma**2)*math.exp(-1/2/sigma**2*(xrange-mu)**2)
    fig = plt.figure()
    p1 = plt.plot(xrange,prior, 'g')
    plt.show()

#True distribution mean and variance 
trueMu = 4
trueVar = 2
LLVar = 20 #Likelihood variance
#Initial prior distribution mean and variance (You should change these parameters to see how they affect the ML and MAP solutions)
priorMu = 8
PMu = priorMu; 
priorVar = 0.2
PVar = priorVar

numDraws = 50 #Number of draws from the true distribution
drawResult=[]


""" ======================  Plot the true distribution ==================== """
#plot true Gaussian function
step = 0.01
l = -20
u = 20
x = np.arange(l+step/2,u+step/2,step)
plt.figure(0)
p1 = plt.plot(x, norm(trueMu,trueVar).pdf(x), color='b')
plt.title('Known "True" Distribution')
MLE = []
MAP = []
"""========================= Perform ML and MAP Estimates =================="""
#Calculate posterior and update prior for the given number of draws
for flip in range(numDraws):
    drawResult.append(np.random.normal(trueMu,np.sqrt(trueVar),1)[0])
    #print(flipResult)
    print('MLE solution for the Gaussian mean' + str(sum(drawResult)/len(drawResult)))
    MLE.append(sum(drawResult)/len(drawResult))
    print('MAP solution for the Gaussian mean' + str(sum(drawResult)*priorVar/(len(drawResult)*priorVar+LLVar)+priorMu*LLVar/(len(drawResult)*priorVar+LLVar)))
    #update the prior Gaussian mean to be replaced with the posterior Gaussian mean from the previous draw
    MAP.append(sum(drawResult)*priorVar/(len(drawResult)*priorVar+LLVar)+priorMu*LLVar/(len(drawResult)*priorVar+LLVar))
    priorMu = sum(drawResult)*priorVar/(len(drawResult)*priorVar+LLVar)+priorMu*LLVar/(len(drawResult)*priorVar+LLVar)  
    #update the prior distribution to be replaced with the posterior distribution from the previous draw
    priorVar = 1/priorVar+len(drawResult)/LLVar
    
"""
You should add some code to visualize how the ML and MAP estimates change
with varying parameters.  Maybe over time?  There are many differnt things you could do!

#plot figure of relationship between MLE, MAP and the number of samples
"""

N = np.arange(0,len(drawResult))
MLE = np.array(MLE)
MAP = np.array(MAP)
plt.figure()
pp2=plt.plot(N,MLE,'o-')
pp1=plt.plot(N,MAP,'o-')
legend=['MLE solution for the Gaussian mean', 'MAP solution for the Gaussian mean']
plt.legend((pp1[0],pp2[0]),legend)
#plt.title('priorMu = %s and priorVar = %s'%(PMu,PVar))
plt.title('Likelihood Variance = %s'%(LLVar))
plt.xlabel('The number of data points')
plt.ylabel('Mean')
plt.savefig('estimation curve')



#Train_2D_class=[]
#Train_2D_class_mean=[]
#Train_2D_class_cov=[]
#PC = []
#for i in range(0,2):
#    Train_2D_class.append(classifytraindata(Train_2D, labels_2D, i, 2))
#    Train_2D_class_mean.append(estimatemean(Train_2D_class))
#    Train_2D_class_cov.append(estimatecov(Train_2D_class))
#    PC.append(Train_2D_class(i)/Train_2D.shape[0])
