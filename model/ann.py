'''
From 
https://medium.com/binaryandmore/beginners-guide-to-deriving-and-implementing-backpropagation-e3c1a5a1e536
Used in regression.py

NB: not used for the training of the article.
'''
#Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1/(1 + np.exp(-x))


#Derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))


def val_appendMatrix(dic, key, matrix):

    valueList = matrix.reshape((np.size(matrix), 1))
    for i in range(len(valueList)):
        if dic[key][i] == None:
            dic[key][i] = valueList[i][0]
        else:
            if not isinstance(dic[key][i] , list):
                dic[key][i] = [dic[key][i]]
            dic[key][i].append(valueList[i][0])

class ANN(object):
    def __init__(self, architecture):
        print('-- Initializating ANN --')
        #architecture - numpy array with ith element representing the number of neurons in the ith layer.
        
        #Initialize the network architecture
        self.L = architecture.size - 1 #L corresponds to the last layer of the network.
        self.n = architecture #n stores the number of neurons in each layer
        #input_size is the number of neurons in the first layer i.e. n[0]
        #output_size is the number of neurons in the last layer i.e. n[L]
        
        #Parameters will store the network parameters, i.e. the weights and biases
        self.parameters = {}
        self.parametersBatch = {}
        #Initialize the network weights and biases:
        for i in range (1, self.L + 1): 
            #Initialize weights to small random values
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            self.parametersBatch['dW' + str(i)] = [None]*np.size(self.parameters['W' + str(i)])
            #Initialize rest of the parameters to 1
            self.parameters['b' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['z' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['a' + str(i)] = np.ones((self.n[i], 1))
            self.parametersBatch['db' + str(i)] = [None]*self.n[i]

        #As we started the loop from 1, we haven't initialized a[0]:
        self.parameters['a0'] = np.ones((self.n[0], 1))
        
        #Initialize the cost:
        self.parameters['C'] = 1
        
        #Create a dictionary for storing the derivatives:
        self.derivatives = {}
                    
    def forward_propagate(self, X):
        #Note that X here, is just one training example
        # print('FORWARD, X', np.shape(X))
        self.parameters['a0'] = X
        
        #Calculate the activations for every layer l
        for l in range(1, self.L + 1):
            # print('FORWARD, W', np.shape(self.parameters['W' + str(l)]))
            # print('FORWARD, a', np.shape(self.parameters['a' + str(l - 1)]))
            # print('FORWARD, b', np.shape(self.parameters['b' + str(l)]))

            self.parameters['z' + str(l)] = np.add(np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)]), self.parameters['b' + str(l)])
            self.parameters['a' + str(l)] = sigmoid(self.parameters['z' + str(l)])
        
    def compute_cost(self, y): # Cross entropy 
        self.parameters['C'] = -(y*np.log(self.parameters['a' + str(self.L)]) + (1-y)*np.log( 1 - self.parameters['a' + str(self.L)]))
    
    def compute_derivatives(self, y):
        #Partial derivatives of the cost function with respect to z[L], W[L] and b[L]:        
        #dC/dzL
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str(self.L)] - y
        #dC/dWL
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))
        #dC/dbL
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        #Partial derivatives of the cost function with respect to z[l], W[l] and b[l]
        for l in range(self.L-1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(np.transpose(self.parameters['W' + str(l + 1)]), self.derivatives['dz' + str(l + 1)])*sigmoid_prime(self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(self.derivatives['dz' + str(l)], np.transpose(self.parameters['a' + str(l - 1)]))
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
           
    def averageBatch(self, dicBatch, parameter, l):
        if parameter == 'dW':
            meanParameter = np.mean(np.array(dicBatch[parameter+str(l)]), axis=1)
            meanParameter = meanParameter.reshape((self.n[l], self.n[l-1]))
        elif parameter == 'db':
            meanParameter = np.mean(np.array(dicBatch[parameter+str(l)]), axis=1)
            meanParameter = meanParameter.reshape((self.n[l], 1))
        return meanParameter         
   
    def update_parameters(self, alpha, option='nobatch'):
        if option=='nobatch':
            for l in range(1, self.L+1):
                self.parameters['W' + str(l)] -= alpha*self.derivatives['dW' + str(l)]
                self.parameters['b' + str(l)] -= alpha*self.derivatives['db' + str(l)]
        elif option=='batch':
            for l in range(1, self.L+1):
                val_appendMatrix(self.parametersBatch, 'dW' + str(l), -alpha*self.derivatives['dW' + str(l)])
                val_appendMatrix(self.parametersBatch, 'db' + str(l), -alpha*self.derivatives['db' + str(l)])
        elif option=='endbatch':
            for l in range(1, self.L+1):
                self.parameters['W' + str(l)] += self.averageBatch(self.parametersBatch, 'dW', l)
                self.parameters['b' + str(l)] += self.averageBatch(self.parametersBatch, 'db', l)
            
    def predict(self, x):
        self.forward_propagate(x)
        return self.parameters['a' + str(self.L)]
    
    def fit(self, X, Y, num_iter, alpha = 0.01):
        print('---> Training ANN ')
        for iter in range(0, num_iter):
            c = 0 #Stores the cost
            n_c = 0 #Stores the number of correct predictions
            
            for i in range(0, X.shape[0]):
                # print('FIT, X', np.shape(X))
                x = X[i].reshape((X[i].size, 1))
                # print('FIT, x', np.shape(x))
                # print('FIT, Y', np.shape(Y))
                y = Y[i]
                # print('FIT, y', np.shape(y))

                self.forward_propagate(x)
                self.compute_cost(y)
                self.compute_derivatives(y)
                self.update_parameters(alpha)
      
                c += self.parameters['C'] 
      
                y_pred = self.predict(x)
                #y_pred is the probability, so to convert it into a class value:
                y_pred = (y_pred > 0.5) 
      
                if y_pred == y:
                    n_c += 1
        print('---> Training ANN completed  ')

    
    def fitBatch(self, X, Y, num_iter, alpha = 0.01, plot=False):
        cost_batch = []
        accuracy_batch = []
        for iter in range(0, num_iter):
            sizeBatch =  10 # Size of the batch
            numberBatch = int(X.shape[0]/sizeBatch) # Total number of batch
            
            for j in range(0, numberBatch):
                Xbatch = X[j:j+numberBatch]
                Ybatch = Y[j:j+numberBatch]
                print('Batch:', j)
                c = 0 #Stores the cost
                n_c = 0 #Stores the number of correct predictions
                for i in range(0, Xbatch.shape[0]):
                    x = Xbatch[i].reshape((Xbatch[i].size, 1))
                    y = Ybatch[i]
                    self.forward_propagate(x)
                    self.compute_cost(y)
                    self.compute_derivatives(y)
                    self.update_parameters(alpha, option='batch')
                    
                    c += self.parameters['C']
                    y_pred = self.predict(x)
                    # print('y', y)
    
                    #y_pred is the probability, so to convert it into a class value:
                    y_pred = (y_pred > 0.5) 
                    # print('ypred', y_pred)
                    if y_pred == y:
                        n_c += 1
                self.update_parameters(alpha, option='endbatch')
                c = c/Xbatch.shape[0]
                acc = (n_c/Xbatch.shape[0])*100
                cost_batch.append(c[0])
                accuracy_batch.append(acc)
                print("Cost: ", c[0])                
                print("Accuracy:", acc)
            
            if plot:
                # Cost
                plt.figure()
                plt.plot(cost_batch)
                plt.xlabel('#batch')
                plt.ylabel('cost')
                plt.title('Total cost per batch')
                plt.figure()                
                plt.plot(accuracy_batch)
                plt.xlabel('#batch')
                plt.ylabel('Accuracy')
                plt.title('Total accuracy per batch')
                
            