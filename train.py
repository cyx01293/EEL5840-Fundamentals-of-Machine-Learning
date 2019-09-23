# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 10:54:29 2018

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

from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from matplotlib.colors import ListedColormap

import torchvision.datasets as dsets
import torchvision.transforms as transforms


# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 32
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

#change the names of training images file and training labels file
imagesTrain = np.load("image_allChar.npy")
labelsTrain = np.load("label_allChar.npy")
"""==========================================================================""" 
"""=======================  Build autoencoder model ========================="""
"""=========================================================================="""
#tranform the numpy value to torch tensor
torchImagesTrain = torch.from_numpy(imagesTrain)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
trainLoader = Data.DataLoader(dataset=torchImagesTrain, batch_size=BATCH_SIZE, shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32, 18*18), #set the encoder layers
            nn.Tanh(), 
            nn.Linear(18*18, 30),   #set the latent layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 18*18),   #deocde from the latent layer
            nn.Tanh(),  
            nn.Linear(18*18, 32*32),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)      #encode the input
        decoded = self.decoder(encoded)  #decode the encoded value
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
lossFunc = nn.MSELoss()

"""==========================================================================""" 
"""=======================  Perform model training==========================="""
"""=========================================================================="""
# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# original data (first row) for viewing
dataView = torchImagesTrain[:N_TEST_IMG].view(-1, 32*32).type(torch.FloatTensor)
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(dataView.data.numpy()[i], (32, 32)), cmap='gray'); 
    a[0][i].set_xticks(()); 
    a[0][i].set_yticks(())

for epoch in range(EPOCH):
    step = 1
    for x, b_label in zip(trainLoader,labelsTrain):
        
        b_x = x.view(-1, 32*32)   # batch x, shape (batch, 32*32)
        b_y = x.view(-1, 32*32)   # batch y, shape (batch, 32*32)
        
    
        encoded, decoded = autoencoder(b_x.float())

        loss = lossFunc(decoded, b_y.float())      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients
        
        step = step + 1
#        print(step)
        if step % 10 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            _, dataDecoded = autoencoder(dataView)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(dataDecoded.data.numpy()[i], (32, 32)), cmap='gray')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); 
            plt.pause(0.05)

plt.ioff()
plt.show()

a = imagesTrain.shape[0]  
#autoencode all training data
dataView = torchImagesTrain[:a].view(-1, 32*32).type(torch.FloatTensor)
dataEncoded, dataDecoded = autoencoder(dataView)
valuesForEncoded = labelsTrain[:a]

"""==========================================================================""" 
"""===========================  Cross validation ============================"""
"""=========================================================================="""
#split the encoded training data to do cross validation
numpyDataEncoded = dataEncoded.data.numpy()
trainX, testX, trainLabel, testLabel = train_test_split(numpyDataEncoded, valuesForEncoded, test_size = 0.1, random_state = 10)

classifiers = []
neighborsN = 6
#KNN classifier is used to classify the encoded value
classifiers.append(neighbors.KNeighborsClassifier(neighborsN, weights='distance'))
names = ['K-NN_Uniform', 'K-NN_Weighted']
for name, clf in zip(names, classifiers):
    clf.fit(trainX, trainLabel)
    score = clf.score(testX, testLabel)
    PredictionsKNN = clf.predict(testX)
AccuracyKNN = accuracy_score(testLabel, PredictionsKNN)
print('\nThe cross validation accuracy of autoencoder and KNN classifier is: ', AccuracyKNN*100, '%')


torch.save(dataEncoded,'dataEncoded.pt')  #save the encoded data as trained model
np.save("valuesForEncoded.npy",valuesForEncoded)  #save the corresponding labels
torch.save(autoencoder.state_dict(), 'autoencoder.pkl') #save the parameters of the model