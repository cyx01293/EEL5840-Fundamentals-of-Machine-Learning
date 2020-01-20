# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:08:27 2018

@author: dataholmes
"""

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from matplotlib import animation
from torch.autograd import Variable
import scipy.misc
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from matplotlib.colors import ListedColormap

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt




LR = 0.005    
#import the trained encoded model
dataEncoded = torch.load('dataEncoded.pt')
valuesForEncoded = np.load("valuesForEncoded.npy")

#import the test data, change file names test images file and test labels file
dataTest = np.load('inData.npy')
#labelTest = np.load('TrainY.npy')

"""==========================================================================""" 
"""=======================  Modify the test images size ====================="""
"""==========================================================================""" 
image = dataTest

imageReshaped = []
for i in range(len(image)):
    im = np.array(image[i])
    im = Image.fromarray(im.astype(np.uint8))
    img = im.resize((32,32))  #the modified image size is 32×32
    img = np.array(img)
    imageReshaped.append(img)

dataTest = np.array(imageReshaped)

"""==========================================================================""" 
"""=======================  Import autoencoder =============================="""
"""==========================================================================""" 
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 18*18),
            nn.Tanh(), 
            nn.Linear(18*18, 30),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 18*18),
            nn.Tanh(),  
            nn.Linear(18*18, 32*32),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
#        m = nn.Dropout(p =0.4)
#        x = m(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()


optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
lossFunc = nn.MSELoss()
"""==========================================================================""" 
"""=======================  Perform data testing============================="""
"""==========================================================================""" 
#import the parameters of trained model
autoencoder.load_state_dict(torch.load('autoencoder.pkl'))
torchDataTest = torch.from_numpy(dataTest)

#encode all the test data
a=dataTest.shape[0]
dataView = torchDataTest[:a].view(-1, 32*32).type(torch.FloatTensor)
torchTestX, dataTestDecoded = autoencoder(dataView)
testX = torchTestX.data.numpy()
testL = labelTest

#use the trained autoencoder model as training data of KNN
np_dataEncoded = dataEncoded.data.numpy()
trainX = np_dataEncoded
trainL = valuesForEncoded

classifiers = []
n_neighbors = 6
#use KNN classifier to classify the encoded value
classifiers.append(neighbors.KNeighborsClassifier(n_neighbors, weights='distance'))
names = ['K-NN_Uniform', 'K-NN_Weighted']
for name, clf in zip(names, classifiers):
    clf.fit(trainX, trainL)
#    score = clf.score(testX, testL)
    PredictionsKNN = clf.predict(testX)
#AccuracyKNN = accuracy_score(testL, PredictionsKNN)
#print('\nThe test accuracy of autoencoder and KNN classifier is: ', AccuracyKNN*100, '%')
np.save('out.npy',PredictionsKNN)