import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image

torch.set_default_tensor_type("torch.DoubleTensor")

#Setting Parameters
input = 1024
hidden = 1024
class_num = 9

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
testLoader = torch.utils.data.DataLoader(torchXtest, batch_size = 10000, shuffle = False)

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

''' --------------------------------------------------------------------------------------------'''
'''                                  Test the MLP network                                       '''
'''---------------------------------------------------------------------------------------------'''

#Load the trained parameters
mlp = MLP(input, hidden, class_num)
mlp.load_state_dict(torch.load('mlp.pkl'))


#Test the model
predictedLabel = [] #predicted labels
for testData in testLoader:
    testData = Variable(testData.view(-1, 32 * 32))
    outputs = mlp(testData)
    _, predicted = torch.max(outputs.data, 1) # Predict the label of test data
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


