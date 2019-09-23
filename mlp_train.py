import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import train_test_split

torch.set_default_tensor_type("torch.DoubleTensor")

#Setting Parameters
input = 1024
hidden = 1024
classNum = 9
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
'''                                  The frame of MLP network                                   '''
'''---------------------------------------------------------------------------------------------'''
# MLP Model
class MLP(nn.Module):
    def __init__(self, input, hidden, classNum):
        super(MLP, self).__init__()
        self.Fc1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU()                   #activation function:Relu
        self.Fc2 = nn.Linear(hidden, classNum)

    def forward(self, x):
        out = self.Fc1(x)
        out = self.relu(out)
        out = self.Fc2(out)
        return out


mlp = MLP(input, hidden, classNum)

''' --------------------------------------------------------------------------------------------'''
'''                                  Train the MLP network                                      '''
'''---------------------------------------------------------------------------------------------'''

# Train the Model
for epoch in range(numEpochs):
    lossFc = nn.CrossEntropyLoss()  # calculate the loss
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learningRate)  # Adam optimizer
    for x, (trainData, labels) in enumerate(trainLoader):

        trainData = Variable(trainData.view(-1, 32*32))
        labels = Variable(labels-1)

        optimizer.zero_grad()
        outputs = mlp(trainData)
        loss = lossFc(outputs, labels)
        loss.backward()   #backpropagation
        optimizer.step()
        #print iteration times as well as the loss
        if (x + 1) % 5 == 0:
            print('Epoch: ', epoch + 1, '/', numEpochs, ', Progress: ', x + 1, '/', 65, ', Loss: ', np.array(loss.data[0]))

''' --------------------------------------------------------------------------------------------'''
'''                                  Validation and accuracy                                    '''
'''---------------------------------------------------------------------------------------------'''

#Validation function
numCorrect = 0
numTotal = 0
for testData, labels in validLoader:
    testData = Variable(testData.view(-1, 32 * 32))
    outputs = mlp(testData)
    _, predicted = torch.max(outputs.data, 1)
    ones = torch.ones(predicted.shape).type(torch.int64)
    predicted = predicted+ones
    numTotal += labels.size(0)
    numCorrect += (predicted.cpu() == labels).sum()

print('Validation accuracy of the model: %d %%' % (100 * numCorrect / numTotal))

''' --------------------------------------------------------------------------------------------'''
'''                                  Save the trained parameter                                 '''
'''---------------------------------------------------------------------------------------------'''
#save mlp model
torch.save(mlp.state_dict(), 'mlp.pkl')

