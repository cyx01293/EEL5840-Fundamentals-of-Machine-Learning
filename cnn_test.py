import torch
import torch.nn as nn
import torchvision.datasets as dsets
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as function
from PIL import Image

torch.set_default_tensor_type("torch.DoubleTensor")

''' --------------------------------------------------------------------------------------------'''
'''                            Setup the test data and reshape the data                         '''
'''---------------------------------------------------------------------------------------------'''

#Load the test data

X = np.load("inData.npy")

#Reshape the image to 32x32
reshaped_image = []

for i in range(len(X)):
    im = np.array(X[i])
    im = Image.fromarray(im.astype(np.uint8))
    img = im.resize((32,32))
    img = np.array(img)
    reshaped_image.append(img)

reshaped_image = np.array(reshaped_image)

reshaped_image = reshaped_image[:,np.newaxis,:,:]

reshaped_image = reshaped_image.astype(np.int64)

#Convert to torch
torchXtest = torch.from_numpy(reshaped_image).type(torch.DoubleTensor)

#Setup test loader
testLoader = torch.utils.data.DataLoader(torchXtest, batch_size = 128, shuffle = False)

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
        #self.Fc3 = nn.Linear(60, 9)

    def forward(self, x):
        x = function.max_pool2d(function.relu(self.Conv1(x)), (2, 2)) #Maxpooling
        x = function.max_pool2d(function.relu(self.Conv2(x)), (2, 2))
        x = x.view(-1, 16*5*5)
        x = function.relu(self.Fc1(x)) #RELU function
        x = self.Fc2(x)
        return x

''' --------------------------------------------------------------------------------------------'''
'''                                  Test the CNN network                                       '''
'''---------------------------------------------------------------------------------------------'''
#Load the trained parameters
cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pkl'))

#Test the model
predictedLabel = [] #predicted labels

for test_data in testLoader:
    test_data = Variable(test_data)
    outputs = cnn(test_data)
    _,predicted = torch.max(outputs.data, 1) # Predict the label of test data
    ones = torch.ones(predicted.shape).type(torch.int64)
    predicted = predicted+ones

    predictedLabel.extend(np.array(predicted).tolist())

for i in range(len(predictedLabel)):
    if predictedLabel[i] == 9:
        predictedLabel[i] = -1

predictedLabel = np.array(predictedLabel)

''' --------------------------------------------------------------------------------------------'''
'''                                  Save the predicted labels                                  '''
'''---------------------------------------------------------------------------------------------'''
# Save the label of test data
np.save("out.npy",predictedLabel)
