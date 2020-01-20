# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.close('all') #close any open plots

"""
===============================================================================
===============================================================================
============================ Question 1 =======================================
===============================================================================
===============================================================================
"""
""" ======================  Function definitions ========================== """

def generateUniformData(N, l, u, gVar):
    '''generateUniformData(N, l, u, gVar): Generate N uniformly spaced data points 
    in the range [l,u) with zero-mean Gaussian random noise with variance gVar'''
    # x = np.random.uniform(l,u,N)
    step = (u-l)/(N);
    x = np.arange(l+step/2,u+step/2,step)
    e = np.random.normal(0,gVar,N)
    t = np.sinc(x) + e
    return x,t

def plotData(x1,t1,x2,t2,x3=None,t3=None,x4=None,t4=None,legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the 
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo') #plot training data
    p2 = plt.plot(x2, t2, 'g') #plot true value
    if(x3 is not None):
        p3 = plt.plot(x3, t3, 'r') #plot training data
    if(x4 is not None):    
            p4 = plt.plot(x4, t4, 'yo')

    #add title, legend and axes labels
    plt.ylabel('t') #label x and y axes
    plt.xlabel('x')
    
    if(x3 is None):
        plt.legend((p1[0],p2[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0]),legend)
    if(x4 is None):
        plt.legend((p1[0],p2[0],p3[0]),legend)
    else:
        plt.legend((p1[0],p2[0],p3[0],p4[0]),legend)
        
    """
    This seems like a good place to write a function to learn your regression
    weights!
    
    """
        

""" ======================  Variable Declaration ========================== """

l = 0 #lower bound on x
u = 10 #upper bound on x
N = 18 #number of samples to generate
gVar = .25 #variance of error distribution
i =  13#regression model order
Etrain=[] #root-mean-square error between values of training data sets and estimated polynomial
Etest=[] #root-mean-square error between values of true data sets and testing data sets
n=[]
""" =======================  Generate Training Data and Testing Data======================= """


"""This is where you should generate a validation testing data set.  This 
should be generated with different parameters than the training data!   """


data_uniform  = np.array(generateUniformData(N, l, u, gVar)).T #generate N training data points
testdata = np.array(generateUniformData(N+40, l, u, gVar)).T  #generate N+30 testing data points
x1 = data_uniform[:,0]
t1 = data_uniform[:,1]
xt = testdata[:,0]
tt = testdata[:,1]

x2 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
t2 = np.sinc(x2) #compute the true function value
ttr = np.sinc(xt) #compute the true function value at testing data sets
    
""" ========================  Train the Model and Test Model============================= """

""" This is where you should test the validation set with the trained model """


def fitdata(x,t,M,la):
    '''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)''' 
    #This needs to be filled in
    X = np.array([x**m for m in range(M+1)]).T
    w = np.linalg.inv(X.T@X+la*np.identity(M+1))@X.T@t
    return w

for M in range(1,i):

    w = fitdata(x1,t1,M,0.1)
    x3 = np.arange(l,u,0.001)  #get equally spaced points in the xrange
    X = np.array([x3**m for m in range(w.size)]).T
    t3 = X@w #compute the predicted value
    Xfit = np.array([x1**m for m in range(w.size)]).T
    tfit = Xfit@w #fitting the polynomial to the training data points
    Xtest = np.array([xt**m for m in range(w.size)]).T
    ttest = Xtest@w #compute the prediceted value of testing data points
 
    Etrain.append(np.sqrt(np.mean((t1 - tfit) ** 2))) #get the root-mean-square error of training data points and estimated polynomial
    Etest.append(np.sqrt(np.mean((ttest - ttr) ** 2))) #get the root-mean-square error between values of true data points and testing data points

    plt.figure()
    plotData(x1,t1,x2,t2,x3,t3,xt,tt,['Training Data', 'True Function', 'Estimated\nPolynomial', 'Testing Data'])
    plt.title('model order = '+str(M))
    plt.savefig('order%s'%(M))

n = np.arange(1,i)  #get equally spaced points with order sets
Etest=np.array(Etest)  #transfer Etest from list to array
Etrain=np.array(Etrain)  #transfer Etrain from list to array

#plot figure of relationship between root-mean-square and model order
plt.figure()
pp2=plt.plot(n,Etest,'ro-')
pp1=plt.plot(n,Etrain,'go-')
legend=['Training Data', 'Testing Function']
plt.legend((pp1[0],pp2[0]),legend)
plt.ylabel('Erms')
plt.xlabel('model order')
#plt.savefig('Erms')