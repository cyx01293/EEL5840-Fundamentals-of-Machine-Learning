# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:44:06 2018

@author: asus
"""

# -*- coding: utf-8 -*-
"""
File:   hw02.py
Author: Yixing Chen
Date:   Sep 23, 2018
Desc:   
    
"""


""" =======================  Import dependencies ========================== """

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from matplotlib.colors import ListedColormap
#plt.close('all') #close any open plots

""" =======================  Import DataSet ========================== """


Train_2D = np.loadtxt('2dDataSetforTrain.txt')
Train_7D = np.loadtxt('7dDataSetforTrain.txt')
Train_HS = np.loadtxt('HyperSpectralDataSetforTrain.txt')

labels_2D = Train_2D[:,Train_2D.shape[1]-1]
labels_7D = Train_7D[:,Train_7D.shape[1]-1]
labels_HS = Train_HS[:,Train_HS.shape[1]-1]

Train_2D = np.delete(Train_2D,Train_2D.shape[1]-1,axis = 1)
Train_7D = np.delete(Train_7D,Train_7D.shape[1]-1,axis = 1)
Train_HS = np.delete(Train_HS,Train_HS.shape[1]-1,axis = 1)

Test_2D = np.loadtxt('2dDataSetforTest.txt')
Test_7D = np.loadtxt('7dDataSetforTest.txt')
Test_HS = np.loadtxt('HyperSpectralDataSetforTest.txt')

""" ======================  Function definitions ========================== """

"""
===============================================================================
===============================================================================
======================== Probabilistic Generative Classfier ===================
===============================================================================
===============================================================================
"""
#Devide train datasets into different classes with the class number, respectively.
def classifytraindata(traindata, labels, classnumber,dimension):
    traindata_class = np.zeros([1,dimension])
    #Train_2D_class0=np.zeros([Train_2D.shape[0],2])
    if dimension == 174: #If the dataset is hyperspectral dataset, make the classnumber +1
        classnumber+=1
    for i in range(traindata.shape[0]):
        if labels[i] == classnumber:
            traindata_class = np.concatenate((traindata_class, traindata[i,:].copy()[np.newaxis,:]),axis=0)
    traindata_class = np.delete(traindata_class, 0, axis=0)  
    return traindata_class
#Estimate the data mean for every column
def estimatemean(data):
    mean = np.mean(data,axis=0) 
    return mean
#Estimate the covariance matrix 
def estimatecov(data):
    cov = np.cov(data.T)
    return cov
#Estimate the prior probability
def estimatepc(classd, data):
    pc = classd.shape[0]/data.shape[0]
    return pc

#       

""" Here you can write functions to estimate the parameters for the training data, 
    and the prosterior probabilistic for the testing data. """


"""
===============================================================================
===============================================================================
============================ KNN Classifier ===================================
===============================================================================
===============================================================================
"""
#No KNN classifier function
""" Here you can write functions to achieve your KNN classifier. """

#figure params
h = .02  # step size in the mesh
figure = plt.figure(figsize=(17, 9))

#set up classifiers
n_neighbors = 5
classifiers = []
#classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='uniform'))
classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
names = ['K-NN_Uniform', 'K-NN_Weighted']
        

""" ============  Generate Training and validation Data =================== """

""" Here is an example for 2D DataSet, you can change it for 7D and HS 
    Also, you can change the random_state to get different validation data """

# Here you can change your data set
Train = Train_7D #change train data set here to Train_2D, Train_7D or Train_HS
labels = labels_7D  #change labels here to labels_2D, labels_7D or labels_HS
dimension = 7  #change the dimension here to 2, 7 or 174
totalclass = 2 #change the totalclass here to 2, 5
Classes = np.sort(np.unique(labels))


# Here you can change M to get different validation data
M = 20
X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
X_train_class = []
class_number = []
for j in range(Classes.shape[0]):
    jth_class = X_train[label_train == Classes[j],:]
    X_train_class.append(jth_class)

#Visualization of first two dimension of your dataSet
for j in range(Classes.shape[0]):
    plt.scatter(X_train_class[j][:,0],X_train_class[j][:,1],cmap=cm_bright)
i = 1
for j in range(X_train.shape[0]):
    plt.scatter(X_valid[:, 0], X_valid[:, 1], marker='+', c=label_valid, alpha=0.6)


    
    
""" ========================  Train the Classifier ======================== """

""" Here you can train your classifier with your training data """

# Initiate the states

TrainPGC = X_train
labelPGC = label_train
TestPGC = X_valid


Train_class=[]
Train_class_mean=[]
Train_class_cov=[]
Train_class_var=[]
PC = []

y1 = []
y2 = []
su1 = 0
su2 = 0
pos1 = []
pos2 = []
#look at the pdf for classes
for i in range(totalclass):
    Train_class.append(classifytraindata(TrainPGC, labelPGC, i, dimension)) #Classify traindata
    Train_class_mean.append(estimatemean(Train_class[i])) #Estimate mean
    Train_class_cov.append(estimatecov(Train_class[i])) #Estimate full covariance matrix
    Train_class_var.append(np.diag(np.diag(estimatecov(Train_class[i])))) #Estimate diagonal covariance matrix
    PC.append(estimatepc(Train_class[i],TrainPGC))
    y1.append(multivariate_normal.pdf(TestPGC, mean=Train_class_mean[i], cov=Train_class_cov[i], allow_singular = True))
    y2.append(multivariate_normal.pdf(TestPGC, mean=Train_class_mean[i], cov=Train_class_var[i], allow_singular = True))


""" ======================== Cross Validation ============================= """


""" Here you should test your parameters with validation data """
#Calculate posterior
for i in range(totalclass):
    su1 = su1 + PC[i]*y1[i]
    su2 = su2 + PC[i]*y2[i]
for i in range(totalclass):
    pos1.append(PC[i]*y1[i]/su1)
    pos2.append(PC[i]*y2[i]/su2)
    
#Classify the validation datasets into classes. Data points are selected into classes with the largest posterior
predictions_PG = np.zeros([TestPGC.shape[0], 1])
for i in range(TestPGC.shape[0]):
    max_value = 0
    max_index = 0
    for j in range(totalclass):
        if max_value < pos1[j][i]:
            max_value = pos1[j][i]
            if dimension == 174:
                max_index = j+1
            else:
                max_index = j
    predictions_PG[i] = max_index

#Classify the validation datasets into classes. Data points are selected into classes with the largest posterior
predictions_PGdiag = np.zeros([TestPGC.shape[0], 1])
for i in range(TestPGC.shape[0]):
    max_value = 0
    max_index = 0
    for j in range(totalclass):
        if max_value < pos2[j][i]:
            max_value = pos2[j][i]
            if dimension == 174:
                max_index = j+1
            else:
                max_index = j
    predictions_PGdiag[i] = max_index

#Classify the validation datasets into classes with KNN classifier
for name, clf in zip(names, classifiers):
    clf.fit(X_train, label_train)
    score = clf.score(X_valid, label_valid)
    predictions_KNN = clf.predict(X_valid)


# The accuracy for your validation data
accuracy_PG = accuracy_score(label_valid, predictions_PG)
print('\nThe accuracy of Probabilistic Generative classifier is: ', accuracy_PG*100, '%')
accuracy_PGdiag = accuracy_score(label_valid, predictions_PGdiag)
print('\nThe accuracy of Probabilistic Generative classifier is: ', accuracy_PGdiag*100, '%')
accuracy_KNN = accuracy_score(label_valid, predictions_KNN)
print('\nThe accuracy of KNN classifier is: ', accuracy_KNN*100, '%')


""" ========================  Test the Model ============================== """

""" This is where you should test the testing data with your classifier """
#Classify test datasets into classes with KNN classifier
for name, clf in zip(names, classifiers):
    #ax = plt.axes()
    #ax = plt.subplot(len(X_train), len(classifiers) + 1, i)
    clf.fit(X_train, label_train)
    score = clf.score(X_valid, label_valid)
    results_KNN = clf.predict(Test_7D) #change the clf.predict() here to Test_2D, Test_7D, Test_HS
    np.savetxt('2dforTest.txt', results_KNN, fmt='%d') #change the target filename here to 2dforTest.txt, 7dforTest.txt, HSforTest.txt
    




