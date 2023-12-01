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
# Date: April 13th, 2021
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada
'''
Readout and training for the binaryModel
'''

#from snoop import snoop
import numpy as np
import warnings 
import random as rnd
from typing import Sequence, Union
from scipy import sparse
from binary_model.regression import ridge_regression, pseudoInverse_regression, torch_regression
from binary_model.regression import ridgePseudoInverse_regression, neuralNetwork_regression
from binary_model.utils import generateRandomArchitecture, generateRandomWeights
from binary_model.utils import instantiateArchitecture, instantiateWeights, checkDim
import torch

class readout():
        
    def __init__(self, 
                 outSize: int,
                 activation='identity', 
                 architecture=None, 
                 Wout=None,
                 bias=None,
                 **kwargs):
        self.outSize = outSize
        self.Aout = None
        self.Wout = None
        # Architecture
        if architecture is not None:
            self.setArchitectureOut(architecture, **kwargs)
        # Weights
        if Wout is not None:
            self.setWeightsOut(Wout, **kwargs) 
        # Bias
        if bias is not None:
            self.b = bias 
        else:
            self.b = 0
        # Activation of the last layer
        self.activationFunction(activation)
        
    @staticmethod
    def sigmoid(x, param=-1):
        return 1/(1+np.exp(param*x))
    
    @staticmethod
    def exponential(x, tau): 
        return np.exp(-x/tau) 
    
    @staticmethod
    def identity(x): 
        return x
    
    @staticmethod
    def relu(x):
        return [max(0, x)]
    
    @staticmethod
    def clamp(x):
        return np.clip(x, 0, 1)
        
    def activationFunction(self, activation):
        # Select activation function
        if activation=='binary':
            self.f = np.heaviside
        if activation=='sigmoid':
            self.f = readout.sigmoid
        if activation=='tanh':
            self.f = np.tanh
        if activation=='identity':
            self.f  = readout.identity
        if activation=='relu':
            self.f = readout.relu
        if activation=='clip':
            self.f = readout.clamp
        
        # Update metadata
        self.metadata.update({'activation':activation})
    
    def setArchitectureOut(self, 
                          architecture: Union[None, list, np.ndarray], 
                          connectivity = 'FC',
                          **kwargs):
        option = kwargs.get('option', 'other')
        if option == 'exclusion':
            # Exclude neuron connected to the input
            Ain = np.array(self.Ain, dtype=object)
            exclusion = np.unique(np.hstack((Ain)))
            kwargs.update({'exclusion':exclusion})
        else:
            exclusion = []
            
        self.outputParameters = {'connectivity':connectivity,
                                 'option':option}
        # Fully Connected
        if connectivity == 'FC': 
            I = kwargs.get('I', 1.0)
            if I<=1:
                numberNeuron = I*self.N
            elif I>1:
                numberNeuron = I
            self.outputParameters.update({'I':I})
            allIndexes = list(set(np.arange(0, self.N)) - set(exclusion))
            self.idxOut = np.array(rnd.sample(allIndexes, min([int(numberNeuron), len(allIndexes)])), dtype=int)
            self.idxOut = np.sort(self.idxOut)
            self.reducedSize = len(self.idxOut)
            architecture = [np.arange(0, self.reducedSize) for _ in range(self.inSize)] # Fully connected
            self.Aout, _ = instantiateArchitecture(architecture, Nin=self.inSize, Nout= self.reducedSize)
        # Partially connected
        elif connectivity != 'PC': # [ToDo]
            # If no architecture is provided
            if architecture is None:
                architecture = generateRandomArchitecture(self, layer='readout', **kwargs)
            type_ = kwargs.get('type', 'array')
            # Initialize architecture list and matrix
            self.Aout, _ = instantiateArchitecture(architecture, Nin=self.outSize, Nout= self.N)
            self.idxOut = np.unique(np.array(np.hstack((self.Aout)), dtype=int))
            self.idxOut = np.sort(self.idxOut)
            self.reducedSize = len(self.idxOut)
            # Reallocate indexes to create the reduced Aout matrix  
            Aout_newIdx = np.array(self.Aout.copy())
            for i, idx in enumerate(self.idxOut):
                Aout_newIdx[Aout_newIdx == idx] = i      
            # Create the reduced Aout matrix (needed for the training)
            self.Aout_matrix = np.zeros((self.outSize, len(self.idxOut)))
            for i in range(self.outSize): 
                self.Aout_matrix[i, Aout_newIdx[i].astype(int)] = 1
            if type_ == 'array':
                pass
            elif type_ == 'sparse':
                self.Aout_matrix = sparse.csc_matrix(self.Aout_matrix).transpose()
            else:
                raise Exception('in "setArchitectureOut()" : type not supported.')   
        else:
            raise Exception('in "setArchitectureOut()" : option not supported')
            
    def setWeightsOut(self, 
                     weights: Union[None, list, np.ndarray], 
                     option='array', 
                     **kwargs):
        # If no weights is provided
        if weights is None:
            weights = generateRandomWeights(self, layer='readout', **kwargs)    
        
        # Arguments for weight initialization
        if self.Aout is not None:
            print('SetWeightOut', 'A is not None')
            kwargs = {'A':self.Aout}
        else:
            raise Exception('Error! Aout must be initialized to set weights')
        if kwargs.get('Nin', None) is None:
            kwargs.update({'Nin':self.outSize})
        if kwargs.get('Nout', None) is None:
            kwargs.update({'Nout':self.reducedSize})
        kwargs.update({'option':option})
        
        # Initialize weights
        self.Wout_matrix , self.Wout, _ = instantiateWeights(weights, **kwargs)
        self.Wout_matrix = self.Wout_matrix.transpose() 
        
    def setBias():
        # [Todo]
        pass
        
    def updateReadout(self, x=None):
        if x is None:
            x = self.x[:, self.idxOut]
        return self.f(x @ self.Wout_matrix + self.b)
    
    def costFunction(self, y_target, option='mse'): # ToDO
        length = np.shape(self.y_concatenated)[1]
        print('y_pred', np.shape(self.y_concatenated))
        y_target, _ = checkDim(y_target, Nin=self.outSize, Nout=length)
        print('YTARGET', np.shape(y_target), type(y_target))
        if option=='mse': # Mean square error
            self.c_concatenated = []
            for n in range(self.outSize):
                self.c_concatenated.append((self.y_concatenated[n]-y_target[n])**2)
        self.c_concatenated = np.array(self.c_concatenated)
        avgCost = sum(self.c_concatenated.sum(axis=1)/length)/self.outSize
        print('Average cost of simulation :', avgCost)
        return avgCost

    def read(self, 
            inputs, 
            duration: int, 
            discardTime=500,
            convergingTime=1000,
            y_target = None,
            reset = True,
            feedback = False,
            tag = '', 
            **kwargs):
        readout = kwargs.get('readout', 'output')
        print(f'---- Read - {readout} ----')
        # Argument for the run        
        optionRun = kwargs.get('optionRun', {'spikeCount':True}).copy()
        # Cost option
        if y_target is not None:
            optionCost = kwargs.get('optionCost', 'mse')
        # 1st run to obtain permanent regime
        if convergingTime > 0:
            print('--- Run convergence ---')
            self.inputCond = False
            self.initRandomState(I=0.2)
            self.inputStream = None
            self.run(convergingTime, **optionRun)
        # 2nd run to submit inputs
        u = inputs[0]; t = inputs[1]
        print('-> Run with inputs ->')
        # Prepare inputs for the simulation
        seed = kwargs.get('seed', None)
        self.setInputIn(u, t, duration=duration, seed=seed, tag=tag)

        # Run with inputs
        print('Read, discardTime', discardTime)
        optionRun.update({'discardTime':discardTime})
        optionRun.update({'savingInterval':kwargs.get('savingInterval', 1)})
        self.run(duration, readout=readout, **optionRun)
        # plotData(self, tag=tag)
        if reset:
            self.reset()

        # Cost # ToDo
        if readout == 'output':
            self.y_concatenated, _ = checkDim(self.y_concatenated, Nin=self.outSize, Nout=duration)
            self.y_concatenated = np.array(self.y_concatenated)
            if y_target is not None:
                self.costFunction(y_target, optionCost)
                return self.y_concatenated[0], self.c_concatenated
            else:
                return self.y_concatenated[0]
    
    def train(self, 
              u_trains: Sequence[np.array], 
              y_targets: Sequence[np.array], 
              duration: int,
              discardTime: int = 500,
              convergingTime: int = 1000,
              option='oneshot', 
              optimizer='pseudoinverse',
              I: float = 1.0,
              tag = None,
              **kwargs):
        print('---- Train readout ----')
        # Update metadata
        self.metadata.update({'learning':option,
                              'optimizer':optimizer,
                              'convergingTime':convergingTime,
                              'discardTime':discardTime})
        # Update kwargs
        kwargs.update({'readout':'train'})
        
        if option=='online':
            
            if optimizer == 'pytorch':
                print('--> pytorch regression')
                optionOptim = kwargs.get('optionOptim').copy()
                activation = optionOptim.get('activation', 'sigmoid')
                optim = optionOptim.get('optim', 'SGD')
                loss = optionOptim.get('loss', 'MSE')
                # Parameters
                self.activationFunction(activation)            
                inSize = self.reducedSize
                outSize = self.outSize
                nbSample = optionOptim.get('nbSample', duration-discardTime)
                # Update metadata
                self.metadata.update({'optionOptim':optionOptim})
                
                # Get the optimizer
                optimizer = torch_regression(inSize, outSize, activation=activation, 
                                             loss=loss, optim=optim)
                
                # Remove argument for the optimizer
                optionOptim.pop('optionRun', None)
                optionOptim.pop('optim', None)
                optionOptim.pop('loss', None)
                optionOptim.pop('activation', None)
                optionOptim.pop('nbSample', None)

                # Run each sample
                layer = None
                for i, (inputs, y_target) in enumerate(zip(u_trains, y_targets)):   
                    self.read(inputs=inputs, duration=duration,
                             discardTime=discardTime, convergingTime=convergingTime,
                             tag=tag,**kwargs)
                    # Check dimensions
                    X = np.array(self.x_concatenated)
                    Yt = np.array(y_targets)
                    X, _ = checkDim(self.x_concatenated, Nin=nbSample, Nout=self.reducedSize)
                    Yt, _ = checkDim(y_target, Nin=nbSample, Nout=self.outSize)
                    X = np.array(X)
                    X = torch.tensor(X, dtype=torch.float32)
                    X = X.view(nbSample, inSize)
                    Yt = np.array(Yt)
                    Yt = torch.tensor(Yt, dtype=torch.float32)
                    Yt = Yt.view(nbSample, outSize)
                    # print('train X', np.shape(X), type(X))
                    # print('train Y', np.shape(Yt), type(Yt))
                    # Train the weights
                    Wout, b, layer = optimizer(X, Yt, layer=layer, **optionOptim)  
                    print('train, W', np.shape(Wout))
                    print('train, b', np.shape(b))
                    
                    # Reset state
                    self.reset()

                self.Wout = Wout
                self.b = b
                self.setWeightsOut(self.Wout)
            
            elif optimizer == 'ANN':
                # Parameters
                self.activationFunction('sigmoid')            
                alpha = kwargs.get('alpha', 0.0001)
                nbiter = kwargs.get('nbiter', 100)
                architecture = np.array([self.reducedSize, self.outSize])
                # Update metadata
                self.metadata.update({'alpha':alpha,
                                      'nbiter':nbiter})
                # Get the optimizer
                optimizer = neuralNetwork_regression(architecture)

                # Run each sample
                Wout, b = None, None
                for i, (inputs, y_target) in enumerate(zip(u_trains, y_targets)):   
                    self.read(inputs=inputs, duration=duration,
                             discardTime=discardTime, convergingTime=convergingTime,
                             tag=tag,**kwargs)
                    # Check dimensions
                    X = np.array(self.x_concatenated)
                    Yt = np.array(y_targets)
                    X, _ = checkDim(self.x_concatenated, Nin=duration-discardTime, Nout= self.reducedSize,)
                    Yt, _ = checkDim(y_target, Nin=duration-discardTime, Nout=self.outSize)
                    # print('train X', np.shape(X), type(X))
                    # print('train Y', np.shape(Yt), type(Yt))
                    self.reset()
                    # Train the weights
                    X = np.array(X, dtype=float)
                    Yt = np.array(Yt, dtype=float)
                    Wout, b = optimizer(X, Yt, nbiter, alpha, Wout, b)  
                self.b = b
                self.Wout = Wout
                self.setWeightsOut(self.Wout)
                
        elif option=='oneshot':
        
            activation = kwargs.get('activation', 'identity')
            self.activationFunction(activation)
            # Select linear model
            if optimizer == 'pseudoinverse':
                modelSolver = pseudoInverse_regression(activation)
            elif 'ridge' in optimizer:
                warnings.warn('oneshot training will use identity activation function')
                beta = kwargs.get('beta', 1.E-8)
                if optimizer == 'ridge':  
                    modelSolver = ridge_regression(beta)
                elif optimizer == 'ridgepseudo':
                    modelSolver  = ridgePseudoInverse_regression(beta)
                self.metadata.update({'beta': beta})
                 
            # Run each sample
            X = []
            for i, (inputs, y_target) in enumerate(zip(u_trains, y_targets)): 
                self.read(inputs=inputs, duration=duration,
                         discardTime=discardTime, convergingTime=convergingTime,
                         tag=tag,**kwargs)
                self.x_concatenated = np.array(self.x_concatenated)
                
                y_target = np.array(y_target)
                # Check dimensions
                X, _ = checkDim(self.x_concatenated, Nin=self.reducedSize, Nout=duration-discardTime)
                Yt, _ = checkDim(y_target, Nin=self.outSize, Nout=duration-discardTime)
                print('train X', np.shape(X), type(X))
                print('train Y', np.shape(Yt), type(Yt))

                X = np.array(X, dtype=float)
                Yt = np.array(Yt, dtype=float)
                # Concatenate all trials
                if i == 0:
                    X_concatenated = X
                    Yt_concatenated = Yt
                else:
                    X_concatenated = np.concatenate((X_concatenated, X), axis=1)
                    Yt_concatenated = np.concatenate((Yt_concatenated, Yt), axis=1)
                self.reset()
            # Train the weights
            self.Wout = modelSolver(X_concatenated, Yt_concatenated)
            self.setWeightsOut(self.Wout)
        print('---- Readout trained ----')
        
    def predict(self,  
                inputs,
                data,
                startTime: int,
                duration: int, 
                delay: int = 1,
                timeBeforePred: int = 1,
                convergingTime: int = 0,
                I: float = 1.0,
                tag= None, 
                **kwargs):
        print('---- Predict ----')
        # Run brefore prediction
        self.read(inputs, 
                   duration=startTime,
                   discardTime=startTime-1, 
                   convergingTime=convergingTime, 
                   reset=False, 
                   tag=tag,
                   **kwargs)
        
        # Input for prediction
        u1 = data[startTime:startTime+timeBeforePred]
        t1 = np.arange(0, timeBeforePred) #t[startTime+delay-1:startTime+delay+timeBeforePred]-t[startTime+delay-1]

        # Prepare inputs for the simulation
        print('predict, u, t', np.shape(u1), np.shape(t1))
        self.setInputIn([u1], [t1], duration=timeBeforePred, tag=tag, **kwargs)
        
        # Last output
        y0 = self.y_concatenated[-1]
        print('u t+1 :', u1[0])
        print('y t+1 :', y0)    
        
        print('predict, u', np.shape(self.inputStream['u']))
        
        # Options
        optionRun = kwargs.get('optionRun', {'spikeCount':True}).copy()
        optionRun.update({'readout':'output'})
        # Predict
        self.feedback = True
        print('duration', duration)
        self.run(duration, read=True, **optionRun)
        self.feedback = False

        print('predict, y', np.shape(self.y_concatenated))

        return self.y_concatenated
                
    
