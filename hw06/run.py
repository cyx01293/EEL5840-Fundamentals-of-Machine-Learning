
import numpy as np
import mlp
from sklearn.model_selection import train_test_split
#load data
data = np.load('dataSet.npy')


#Set up Neural Network
data_in = np.copy(data)
data_in = np.delete(data_in,2,axis=1)
target_in = np.copy(data)
target_in = np.delete(target_in,0,axis=1)
target_in = np.delete(target_in,0,axis=1)
hidden_layers = 3 #change number of nodes on hidden layer here
X_train, X_valid, target_train, target_valid = train_test_split(data_in, target_in, test_size = 0.33, random_state = 10)
NN = mlp.mlp(data_in,target_in,hidden_layers)
eta = 0.1
enter = np.zeros((1,2))
error = np.zeros((data_in.shape[0],1))

#errorr=mlp.mlp.earlystopping(NN,X_train,target_train,X_valid,target_valid,eta,niterations=100)
for i in range(data_in.shape[0]):
    enter[0,:] = data_in[i,:]
    group = target_in[i]
    #NN = mlp.mlp(enter,group,hidden_layers)
    error[i] = mlp.mlp.earlystopping(NN,X_train,target_train,X_valid,target_valid,eta,niterations=100)
    
#Analyze Neural Network Performance
mlp.mlp.confmat(NN,X_valid,target_valid)