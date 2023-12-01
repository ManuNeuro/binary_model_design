# Copyright (c) 2012-2021, NECOTIS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Authors: Emmanuel Calvet, Jean Rouat (advisor), Bertrand Reulet (co-advisor)
# Date: April 14th, 2021
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada
'''
regression model that can be used for training the readout

NB: not used for the final version of the paper
'''
import numpy as np
from sklearn import svm
from binary_model.ann import ANN
from binary_model.torchoptimizer import torchoptimizer

def torch_regression(inSize, outSize, 
                     activation='sigmoid',  loss='MSE', 
                     optim='SGD'):

    model = torchoptimizer(inSize, outSize, 
                           activation, loss, 
                           optim)
    
    def gradient_descent(X, Y, layer=None, **kwargs):    
        if layer is not None:
            model.linear = layer
        model.train(X, Y, **kwargs)
        return model.w, model.b, model.linear
    
    return gradient_descent

def ridge_regression(beta=1.E-08):
    
    def ridge_modelSolver(X, Y):
        return Y @ X.T @ np.linalg.inv(X @ X.T + beta * np.eye(X.shape[0]))
    
    return ridge_modelSolver
    
def pseudoInverse_regression(activation='identity'):
    
    def pseudoInverse_modelSolver(X, Y):
        if activation=='sigmoid':
            return (-np.log(1-Y)+np.log(Y)) @ np.linalg.pinv(X)
        if activation=='identity' or activation=='clip':
            print('pseudoInv, Y, X', np.shape(Y), np.shape(X))
            return Y @ np.linalg.pinv(X)
    
    return pseudoInverse_modelSolver

def ridgePseudoInverse_regression(beta=1.E-08):
    
    def ridgePseudoInverse_modelSolver(X, Y):
        return Y @ X.T @ np.linalg.pinv(X @ X.T + beta * np.eye(X.shape[0]))
    
    return ridgePseudoInverse_modelSolver


def neuralNetwork_regression(architecture):
    model = ANN(architecture)
    
    def gradientDescent_modelSolver(X, Yt, nbiter=100, alpha=0.1, Wout=None, b=None):
        if Wout != None:
            model.parameters['W1'] = Wout
        if b != None:
            model.parameters['b1']  = b
        model.fit(X, Yt, nbiter, alpha)
        return model.parameters['W1'], model.parameters['b1'] 
    
    return gradientDescent_modelSolver

# ToDo : not working properly
def backpropagation_regression(alpha=0.01, cost='mse', activation='relu'):
    
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def backpropagation_modelSolver(X, Y, Yt):            
        # Cost derivative
        if cost == 'mse':     
            dc_dy = 2. * (Y - Yt)  # dC/dz, MSE as cost     
        elif cost == 'cross-entropy' and activation == 'sigmoid':
            dc_dz = (Y - Yt)
        
        # Activation derivative
        if activation == 'relu':            
            dy_dz = Y.copy(); dy_dz[dy_dz<=0]=0; dy_dz[dy_dz>0]=1; # dz/dy
            dc_dz = dc_dy * dy_dz
        elif activation == 'identity':
            dy_dz = 1
            dc_dz = dc_dy * dy_dz
        elif cost != 'cross-entropy' and activation == 'sigmoid':
            dy_dz = sigmoid(Y)*(1-sigmoid(Y))
            dc_dz = dc_dy * dy_dz
            
        # Updating parameters
        deltaW = -alpha * dc_dz @ X # dC/dW
        deltaB = -alpha * dc_dz # dC/db       
        return deltaW, deltaB

    return backpropagation_modelSolver

# ToDo
def svm_regression(kernel='linear', **kwargs):

    def svm_modelSolver(X, Y):
        degree =  kwargs.get('degree', 3)
        gamma =  kwargs.get('gamma', 'auto')   
        C = kwargs.get('C', 1.0)
        clf = svm.SVC(C, kernel, degree, gamma) # Kernel
        #Train the model using the training sets
        return clf.fit(X, Y)
    
    return svm_modelSolver
