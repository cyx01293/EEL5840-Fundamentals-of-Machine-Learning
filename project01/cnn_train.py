import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as function

torch.set_default_tensor_type("torch.DoubleTensor")

#Setting Parameters
numEpochs = 15
batchSIZE = 128
learningRate = 0.01

''' --------------------------------------------------------------------------------------------'''
'''         Setup the train data, validation data, train labels and validation labels           '''
'''---------------------------------------------------------------------------------------------'''

#Setup train dataset
X = np.load("X.npy")
Y = np.load("Y.npy")


X = X.astype(np.int64)
Y = Y.astype(np.int64)

X  = X[:,np.newaxis,:,:]

# Split the train data and valid data
trainData, validData, trainLabel, validLabel = train_test_split(X, Y, test_size=.2)

# Covert the numpy array to tensor
torchXtrain = torch.from_numpy(trainData).type(torch.DoubleTensor)
torchYtrain = torch.from_numpy(trainLabel)

torchXvalid = torch.from_numpy(validData).type(torch.DoubleTensor)
torchYvalid = torch.from_numpy(validLabel)

#Combine the data and labels
train = torch.utils.data.TensorDataset(torchXtrain,torchYtrain)
valid = torch.utils.data.TensorDataset(torchXvalid,torchYvalid)

#Setup the loader for batch training and validation
trainLoader = torch.utils.data.DataLoader(train, batch_size = batchSIZE, shuffle = True)
validLoader = torch.utils.data.DataLoader(valid, batch_size = batchSIZE, shuffle = True)

''' --------------------------------------------------------------------------------------------'''
'''                                  The frame of CNN network                                   '''
'''---------------------------------------------------------------------------------------------'''
#CNN network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 8, 5) #Convolution layer
        self.Conv2 = nn.Conv2d(8, 16, 5)

        self.Fc1   = nn.Linear(16*5*5,100)
        self.Fc2   = nn.Linear(100, 9) #Hidden layer

    def forward(self, x):
        x = function.max_pool2d(function.relu(self.Conv1(x)), (2, 2)) #Maxpooling
        x = function.max_pool2d(function.relu(self.Conv2(x)), (2, 2))
        x = x.view(-1, 16*5*5)
        x = function.relu(self.Fc1(x)) #RELU function
        x = self.Fc2(x)
        return x

''' --------------------------------------------------------------------------------------------'''
'''                                  Train the CNN network                                      '''
'''---------------------------------------------------------------------------------------------'''

#Train function
def Train():
    lossFc = nn.CrossEntropyLoss() #Calculate the cross entropy loss
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learningRate) #Adam optimizer
    cnn.eval()
    for epoch in range(numEpochs):
        for x, (images, labels) in enumerate(trainLoader):
            images = Variable(images)
            labels = Variable(labels-1)

            optimizer.zero_grad()
            outputs = cnn(images) #Forward
            loss = lossFc(outputs, labels)
            loss.backward() #Backward
            optimizer.step() #Optimize

            if (x+1) % 5 == 0:
                print('Epoch: ',epoch + 1,'/',numEpochs, ', Progress: ', x + 1,'/', len(trainData) // batchSIZE+1, ', Loss: ', np.array(loss.data[0]))

''' --------------------------------------------------------------------------------------------'''
'''                                  Validation and accuracy                                    '''
'''---------------------------------------------------------------------------------------------'''
#Validation function
predictedLabel = [] #Predicted label
validLabel = []     #True label

def Validation():
    numCorrect = 0
    numTotal = 0
    cnn.eval()
    for images, labels in validLoader:
        images = Variable(images)
        outputs = cnn(images)

        _, predicted = torch.max(outputs.data, 1) # Predict the label of valid data
        ones = torch.ones(predicted.shape).type(torch.int64)
        predicted = predicted+ones

        numCorrect += (predicted.cpu() == labels).sum()
        numTotal += labels.size(0)
        labels = labels.type(torch.int64)

        predictedLabel.extend(np.array(predicted).tolist())
        validLabel.extend(np.array(labels).tolist())

    print('The predicted label is :', predictedLabel)
    print('The true label is :', validLabel)

    print('Validation accuracy of the model : %d %%' % (100 * numCorrect / numTotal))

#Train the model
cnn = CNN()
Train()
Validation()

''' --------------------------------------------------------------------------------------------'''
'''                                  Save the trained parameter                                 '''
'''---------------------------------------------------------------------------------------------'''
#Save the trained CNN parameters
torch.save(cnn.state_dict(), 'cnn.pkl')