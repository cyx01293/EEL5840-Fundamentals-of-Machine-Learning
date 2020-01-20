# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
#        self.weights1=np.array([[1,0,1.5],[1,0,2.75],[1,0,3.5]])
#        self.weights2=np.array([[1],[-0.5],[1],[0]])

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        #valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
        #while(count<niterations):
            count+=1
            print(count)
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing - ADD CODE HERE """    
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        deltaw1 = np.zeros((np.shape(self.weights1)))
        deltaw2 = np.zeros((np.shape(self.weights2)))
        
        
        for i in range(niterations):
            self.y = self.mlpfwd(inputs)
            error = targets - self.y
            #gradientw2 = (error)*(-1)*(self.y)*(1-self.y)*np.transpose(self.hidden)
            gradientw2 = (error)*(-1)*(self.y)*(1-self.y)
            #gradientw1 = self.hidden*(1-self.hidden)*np.dot((error)*(-1)*(self.y)*(1-self.y),np.transpose(self.weights2))
            gradientw1 = np.dot((error)*(-1)*(self.y)*(1-self.y),np.transpose(self.weights2))*self.hidden*(1-self.hidden)
            
            deltaw2 = np.dot(np.transpose(self.hidden),eta*gradientw2)+ self.momentum*deltaw2
            #deltaw2 = np.dot(np.transpose(self.hidden),eta*gradientw2)+ self.momentum*deltaw2
            deltaw1 = eta*np.dot(np.transpose(inputs),(gradientw1[:,:-1]))+ self.momentum*deltaw1
            #deltaw1 = eta*np.dot(np.transpose(gradientw1[:,:-1]),(inputs)) + self.momentum*deltaw1
            self.weights2 = self.weights2 - deltaw2
            self.weights1 = self.weights1 - deltaw1 
            
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)


