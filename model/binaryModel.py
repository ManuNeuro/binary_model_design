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
# Date: March 16th, 2021
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada

import numpy as np
import random as rnd
from scipy import sparse
import os
from typing import Union
from binary_model.utils import val_append, checkStateDim, splitpath
from binary_model.utils import instantiateArchitecture, instantiateWeights
from binary_model.inputlayer import inputLayer
from binary_model.readout import readout

def param_names(name):
    dic = {'meanWeights':'mu', 
           'stdWeights':'std'}
    return dic.get(name, name)

class binaryModel(inputLayer, readout):
    '''
    Binary neural model adapted from
    [Natschlager, Bertschingerm Legenstein, 2004].
    This model is capable of uni and multidimensional inputs, and 
    display intersting phase transition as controlled by  
    connectivity and synaptic weights parmeters.
    
    Parameters __init__
    ----------
    - N : int
        number of neurons
    - K : int or None
        1. None : (default)
            The network architecture will not be generated.
        2. int :
            The network architecture will be generated with connectivity
            degree K, via initRandomArchitecture(K).
    - experiment : str
        1. '' : (default)
            No impact
        2. str :
            Name of the experiment will be added in metadata, along the 
            creation of recipient folder for simulation results.
    - architecture : None or np.array, or list
        1. None : (default)
            No architecture is provided to the network.
        2. np.array or list :
            An architecture is provided to the network and it will be
            initialized with setArchitecture(architecture).
    - weights : None or np.array, or list.
        1. None : (default)
            No weights matrix is provided to the network.
        2. np.array or list :
            Synaptic weights will be initialized with 
            setWeights(weights).
    '''
    def __init__(self, N=None, K=None, sim='', experiment='', architecture=None, weights=None):
        # Parameter of the network
        self.N = N
        self.inSize = 1
        self.x = np.zeros((1, N))
        self.A = None
        self.inputCond = False
        self.inputLayer = False
        self.inputcurrent = False
        # Parameter of the simulation and data 
        self.c = 0
        self.inputStream = None
        self.feedback = False
        self.nbTrials = 0
        self.duration = 0
        self.spikeCount = []
        self.spikeCountTrials = []
        self.spikeDiagram = []
        self.activeIndex = []
        self.activeIndexTrials = []
        self.outputTrials = []
        self.continuousState = False
        # Parameter of metatadata and files
        self.fileSpec = None
        self.metadata = {}
        self.experiment = experiment
        self.sim = sim
        self.seed = {}

        # Init architecture
        if (K is not None) and (architecture is None):
            self.initRandomArchitecture(K)

        elif (K is None) and (architecture is not None):
            self.setArchitecture(architecture)

        # Init weights
        if (weights is not None):
            self.setWeights(weights)

        # metadata
        self.metadata.update({'N': N})


    ###########################################################################
    # Methods for the architecture and weights
    ###########################################################################
    def setArchitecture(self, architecture: Union[list, np.ndarray]):
        
        # Initialize architecture
        self.A, self.K = instantiateArchitecture(architecture, Nin=self.N, Nout=self.N)
              
        # metadata
        self.metadata.update({'K': self.K})

    def setWeights(self, weights, option='sparse'):
        # Arguments
        if self.A is not None:
            print('SetWeightOut', 'A is not None')
            kwargs = {'A':self.A}
        else:
            raise Exception('Error! A must be initialized to set weights')
        kwargs.update({'Nin':self.N,
                       'Nout':self.N,
                       'option':option})
        # Initialize weights
        self.W_matrix, self.W, statistics = instantiateWeights(weights, **kwargs)
        self.meanWeights = statistics[0]
        self.stdWeights = statistics[1]
        # metadata
        self.metadata.update({'meanWeights': self.meanWeights,
                              'stdWeights': self.stdWeights})

            
    ###########################################################################
    # Methods for adding layers
    ###########################################################################
    def addInputLayer(self, inSize=0, weights=None, architecture=None, seed=None, tag=None, **kwargs):
        # Init input class
        if weights == 'auto':
            weightsInit = None
        else:
            weightsInit = weights
        if architecture == 'auto':
            architectureInit = None
        else:
            architectureInit = architecture
        inputLayer.__init__(self, self.N, inSize, weightsInit, architectureInit)
        
        # Generate architecture and weights
        if architecture == 'auto':
            option = kwargs.get('option', 'global')
            I = kwargs.get('I', 1.0)
            self.setArchitectureIn(None, option=option, I=I, seed=seed)   
        if weights  == 'auto':
            distribution = kwargs.get('distribution', 'uniform')
            kwargs.pop('distribution', None)
            self.setWeightsIn(None, distribution=distribution, seed=seed, **kwargs)
        # Update metadata
        self.metadata.update({'input':tag})
    
    def addReadoutLayer(self, outSize, activation='identity', 
                        architecture=None, weights=None, bias=None, 
                        seed=None, tag=None, **kwargs): 
        #architecture = kwargs.get('architecture', 'SOUCE')
        print('addReadoutLayer', activation, architecture)
        # Init readout class
        if architecture == 'auto':
            architectureInit = None
        else:
            architectureInit = architecture
        if weights == 'auto':
            weightsInit = None
        else:
            weightsInit = weights
        readout.__init__(self, outSize, activation, architectureInit, weightsInit, bias, **kwargs)
        # Generate architecture and weights
        print('addReadoutLayer', activation, architecture)
        if architecture == 'auto':
            option = kwargs.get('option', 'global')
            kwargs.pop('option', None)
            self.setArchitectureOut(None, option=option, seed=seed, **kwargs)   
        if weights  == 'auto':
            distribution = kwargs.get('distribution', 'uniform')
            kwargs.pop('distribution', None)
            self.setWeightsOut(None, distribution=distribution, seed=seed, **kwargs)
        # Update metadata
        self.metadata.update({'readout':tag})
        
    ###########################################################################
    # Methods for initializing the network
    ###########################################################################        
    def initRandomState(self, I=None, x=None, seed=None, indexes=None, **kwargs):
        
        if x is None:
            self.setSeedNetwork(seed, tag='initialState')
            if indexes is None:
                nI = int(self.N * I)  # Number of ones
                x0 = np.zeros(self.N);
                x0[:nI] = 1
                np.random.shuffle(x0)
                self.x = np.array([x0], dtype=int)
            else:
                indexes = np.array(indexes, dtype=int)
                nI = int(len(indexes) * I)  # Number of ones
                x0 = np.zeros(len(indexes)) # Generate zeros
                self.x = np.zeros((1, self.N)) # reset state
                x0[:nI] = 1
                np.random.shuffle(x0)
                self.x[0, indexes] = np.array(x0, dtype=int)
        elif x is not None:
            if len(np.shape(x)) != 2:
                self.x = [x]
            else:
                self.x = x
            checkStateDim(self.x, self.N)
            
        spikeCount = kwargs.get('spikeCount', True)
        indexActive = kwargs.get('indexActive', False)
        spikeDiagram = kwargs.get('spikeDiagram', False)
        if spikeCount:
            xSpike = self.x[self.x == 1]
            self.spikeCount.append(len(xSpike) / self.N)
        if indexActive:
            idxActive = sparse.csr_matrix(self.x).indices  # Find non zero indexes
            self.activeIndex = np.unique(np.hstack((self.activeIndex, idxActive)))
        if spikeDiagram:
            self.spikeDiagram = np.vstack((self.spikeDiagram, self.x))
        self.metadata.update({'I': I})
        
    def setInputCurrent(self, c, t, tag='None'): 
        # [ToDo] : simplify and check dimensions
        '''
        Set the input of the network, two vectors
        are needed, one for timings, and one for amplitude of
        excitation.

        Parameters
        ----------
        c : np.array or list of list
            - 1-D input : Each index is the amplitude associated
            with the same timing index.
            - N-D input : Each index of the list is the input vector of the 
            network associated with the same timing index.
        t : 1-D np.array :
            Timings of the inputs.
        '''
        self.inputCond = True
        self.inputcurrent = True
        # Check if u is 1-D or N-D
        sizeC = np.shape(c)
        if len(sizeC)==1:  # 1-D case
            if len(c)==1 and len(t)>1: # if u is constant
                if isinstance(c[0], type(self.W_matrix)):
                    c = np.array([c*np.ones(len(t))])
            elif len(c)>1 and len(c)!=len(t):
                raise Exception('In "setInputCurrent()" : length of u must be one, or equal to length of t')
            else: # Get a matrix
                c = np.array([c])
                if c.shape[0] == self.N:
                    c = c.T
        if len(sizeC)==2:  # N-D case
            #u = np.array(u, dtype=object)
            if sizeC[0] != self.N:
                raise Exception('In "setInputCurrent()" : size 1 of input current vector u should be N')
            if sizeC[1] != len(t):
                raise Exception('In "setInputCurrent()" : size 2 of input current vector u should be the same as t')
        
        self.inputStream = {'c': c, 't': t}
        
        # Update metadata
        if tag is not None:
            self.metadata.update({'input':tag})    
        
    ###########################################################################
    # Methods for the simulation
    ###########################################################################
    def updateStateNetwork(self, t, **kwargs):
        spikeCount =  kwargs.get('spikeCount', True) 
        indexActive = kwargs.get('indexActive', False)
        spikeDiagram = kwargs.get('spikeDiagram', False)
        condIdxActive = kwargs.get('condIdxActive', 1000) 
        find_gcc = kwargs.get('find_gcc', False)
        inputs = kwargs.get('inputs', False)
        readout = kwargs.get('read', 'None')
        # Update rule
        if not inputs:
            if (find_gcc == True) and (t>=500):
                membranePotential = np.array(self.x @ self.W_matrix + 100*self.x, dtype=float)
            else:
                membranePotential = np.array(self.x @ self.W_matrix, dtype=float)
        else:
            membranePotential = np.array(self.x @ self.W_matrix + self.u @ self.Win_matrix + self.c, dtype=float)
        self.x = np.array(np.heaviside(membranePotential, 0.0), dtype=int)
        
        # if self.continuousState:
        #     self.x_continuous.append(self.x)
        
        # If readout
        if readout == 'analog':
            self.x_analog = self.sigmoid(membranePotential)
            
        # Saving data
        if spikeCount:
            xSpike = self.x[self.x == 1]
            self.spikeCount.append(len(xSpike) / self.N)
            #print(self.spikeCount)
        if indexActive and t >= condIdxActive:
            idxActive = sparse.csr_matrix(self.x).indices
            self.activeIndex = np.unique(np.hstack((self.activeIndex, idxActive)))
        if spikeDiagram:
            self.spikeDiagram = np.vstack((self.spikeDiagram, self.x))
    
    #@snoop
    def run(self, duration, readout='None', **kwargs):
        print('---- RUN ! ----')
        spikeCount = kwargs.get('spikeCount', True)
        indexActive = kwargs.get('indexActive', False)
        if indexActive:
            trials = kwargs.get('trials', True)
            
        # Check if an input has been set
        if self.inputCond:
            print('-> Input sent to the network ->')
            T0 = self.inputStream['t'][0]  # Starting time of stimulus
            Tend = self.inputStream['t'][1]# Ending time of stimulus
            tIdx = 0 # Idx of stimulus time
        
        # If a readout is used
        if readout == 'train' or  readout == 'output':
            self.x_concatenated = []
            if readout == 'output':
                self.y_concatenated = []
            discardTime = kwargs.get('discardTime', 0)
            Ttrain = kwargs.get('Ttrain', duration)
            savingInterval = kwargs.get('savingInterval', 0)
            # if savingInterval > 0:
            #     duration = duration+1 # To access last indexes for saving X and Y states
            print('RUN, discardTime', discardTime)
        # RUN !
        if self.continuousState:
            self.x_continuous = []
        self.duration = duration
        for t in range(duration):
            # Input feedback from readout 
            if self.feedback and not self.inputCond: # [Disclaimer] Not used in article
                kwargs.update({'inputs':True}) 
                self.u = self.y_concatenated[-1] # ToDo
                if len(np.shape(self.u)) <= 1:
                    self.u = self.u.reshape(self.inSize, 1)
                self.inputStream = None
            # Input from inputlayer
            elif self.inputCond:  # Check if the time correspond to input timings
                if T0 <= t <= Tend:
                    kwargs.update({'inputs':True}) 
                    if self.inputLayer:
                        self.u = self.inputStream['u'][tIdx, :]  # update input
                        if self.inputStream['type'] == 'array':
                            self.u = [self.u] # Put in vector
                        elif self.inputStream['type'] == 'sparse':
                            pass
                    if self.inputcurrent: # [Disclaimer] Not used in article
                        self.c = self.inputStream['c'][:, tIdx]  # update input current
                        if self.c != 0:
                            if isinstance(self.c[0], type(sparse.csc_matrix(1))):
                                self.c = self.c[0]
 
                    tIdx += 1  # increment idx timing 
                elif t < T0:
                    kwargs.update({'inputs':False})
                elif t > Tend:
                    kwargs.update({'inputs':False})
                    self.inputCond = False    
                    
            # Update network state
            checkStateDim(self.x, self.N)  # Insure proper dimension of network state
            self.updateStateNetwork(t, **kwargs)
            
            # Update readout state
            if readout != 'None':
                # Case to save states at some regular interval
                if savingInterval > 0:
                    t = t+1 # Trick to align correctly
                    if t>=discardTime+savingInterval and t<=Ttrain:
                        if (t-discardTime) % savingInterval == 0:
                            if readout == 'output':
                                self.y_concatenated.append(self.updateReadout()[0]) # readout state
                            self.x_concatenated.append(self.x[0, self.idxOut]) # reservoir state
                # Case to save all states 
                else:
                    if t>=discardTime and t<=Ttrain:
                        if readout == 'output':
                            self.y_concatenated.append(self.updateReadout()[0]) # readout state
                        self.x_concatenated.append(self.x[0, self.idxOut]) # reservoir state
    
        # Save data
        if spikeCount:
            self.spikeCountTrials.append(list(self.spikeCount))
        if indexActive:
            if trials: # Save indexes of each trial
                self.activeIndexTrials.append(list(self.activeIndex))
            else: # Only save indexes for all trials
                self.activeIndexTrials = np.unique(np.hstack((self.activeIndexTrials, self.activeIndex)))

        self.nbTrials += 1
        
        # Readout, save data
        if readout != 'None':
            if readout == 'output':
                self.outputTrials.append(list(self.y_concatenated))
                return self.x_concatenated, self.y_concatenated
            elif readout == 'train':
                return self.x_concatenated
        
    ###########################################################################
    # Methods for saving datas in file
    ###########################################################################
    def saveNetwork(self, path=None, addPath='', tag='', saveWeights=True, saveArchitecture=True, parameters=None):

        # Save
        if parameters is None:
            self.fileSpec = 'N{0}_K{1}_W{2}_std{3}.npy'.format(self.N, self.K, self.meanWeights, self.stdWeights)
        else:
            fileSpec = f'N{self.N}_K{self.K}'
            for param_name in parameters:
                value = self.metadata.get(param_name)
                param_name = param_names(param_name)
                fileSpec = fileSpec + f'_{param_name}{str(value)}'
            self.fileSpec = fileSpec + '.npy'
            print(self.fileSpec)

        if path is None:
            script_dir = os.path.dirname(__file__) + '/'
            script_dir = splitpath(script_dir, 'model')
            script_dir = script_dir + '/results/'+self.sim
            folder = os.path.join(self.experiment, addPath)
            if parameters is None:
                folder = folder + '/N{0}_K{1}/mu{2}/std{3}/'.format(self.N, self.K, self.meanWeights,
                                                                    self.stdWeights)
            else:
                folder = folder + f'/N{self.N}_K{self.K}/'
                for param_name in parameters:
                    value = self.metadata.get(param_name)
                    param_name = param_names(param_name)
                    folder = folder + f'/{param_name}{str(value)}/'
            path = os.path.join(script_dir, folder)
            if not os.path.isdir(path):
                os.makedirs(path)
        # metadata
        self.metadata.update({'connectivityPath': path})

        if tag != '':
            tag = tag + '_'
        if saveWeights:
            filenameWeight = 'weights_' + tag + self.fileSpec
            pathW = os.path.join(path, filenameWeight)
            file = open(pathW, "wb")
            np.save(file, self.W)
            # metadata
            self.metadata.update({'weihgtsFile': filenameWeight})
        if saveArchitecture:
            filenameArchitecture = 'architecture_' + tag + self.fileSpec
            pathA = os.path.join(path, filenameArchitecture)
            file = open(pathA, "wb")
            np.save(file, self.A)
            # metadata
            self.metadata.update({'architectureFile': filenameArchitecture})

    def saveData(self, path=None, addPath='', tag='', spikeCount=True, indexActive=False, output=False, parameters=None):
        # Filename
        if parameters is None:
            self.fileSpec = 'N{0}_K{1}_D{2}_T{3}_W{4}_std{5}'.format(self.N, self.K, self.duration, self.nbTrials,
                                                                     self.meanWeights, self.stdWeights)
        else:
            self.fileSpec = 'N{0}_K{1}_D{2}_T{3}'.format(self.N, self.K, self.duration, self.nbTrials)
            for param_name in parameters:
                value = self.metadata.get(param_name)
                param_name = param_names(param_name)
                self.fileSpec = self.fileSpec + f'_{param_name}{str(value)}'
                    
        if tag != '':
            tag = tag + '_'
        # Path
        if path is None:
            script_dir = os.path.dirname(__file__) + '/'
            script_dir = splitpath(script_dir, 'model')
            script_dir = script_dir + '/results/'+self.sim
            folder = os.path.join(self.experiment, addPath)
            if parameters is None:
                folder = folder + '/N{0}_K{1}/mu{2}/std{3}'.format(self.N, self.K, self.meanWeights,
                                                                   self.stdWeights)
            else:
                folder = folder + f'/N{self.N}_K{self.K}/'
                for param_name in parameters:
                    value = self.metadata.get(param_name)
                    param_name = param_names(param_name)
                    folder = folder + f'/{param_name}{str(value)}/'
            path = os.path.join(script_dir, folder)
        if not os.path.isdir(path):
            os.makedirs(path)
        if spikeCount:
            print('SAVE', path)
            filenameSpikeCount = 'spikeCount_' + tag + self.fileSpec
            pathSpikeCount = os.path.join(path, filenameSpikeCount)
            np.save(pathSpikeCount+'.npy', self.spikeCountTrials)
            self.metadata.update({'spikeCountFile': filenameSpikeCount})
        if indexActive:
            filenameIdxActive = 'idxActive_' + tag + self.fileSpec
            pathIdxActive = os.path.join(path, filenameIdxActive)
            np.save(pathIdxActive+'.npy', self.activeIndexTrials)
            self.metadata.update({'idxActiveFile': filenameIdxActive})
        if output:
            filenameOutput = 'output_' + tag + self.fileSpec
            pathOutput = os.path.join(path, filenameOutput)
            np.save(pathOutput+'.npy', self.outputTrials)
            self.metadata.update({'outputFile': filenameOutput})
        # metadata
        self.metadata.update({'dataPath': path})

    def saveMetadata(self, path=None, addPath='', tag='', parameters=None):

        # Filename
        if self.fileSpec is None:
            if parameters is None:
                self.fileSpec = 'N{0}_K{1}_D{2}_T{3}_W{4}_std{5}'.format(self.N, self.K, self.duration, self.nbTrials,
                                                                         self.meanWeights, self.stdWeights)
            else:
                self.fileSpec = 'N{0}_K{1}_D{2}_T{3}'.format(self.N, self.K, self.duration, self.nbTrials)
                for param_name in parameters:
                    value = self.metadata.get(param_name)
                    self.fileSpec = self.fileSpec + f'_{param_name}{str(value)}'
            
        if tag != '':
            tag = tag + '_'
        filenameMetadata = 'metadata_' + tag + self.fileSpec

        # Path
        if path is None:
            script_dir = os.path.dirname(__file__) + '/'
            script_dir = splitpath(script_dir, 'model')
            script_dir = script_dir + '/results/'+self.sim
            folder = os.path.join(self.experiment, addPath)
            if parameters is None:
                folder = folder +'/N{0}_K{1}/mu{2}/'.format(self.N, self.K, self.meanWeights)
            elif parameters == 'None':
                folder = folder + f'/N{self.N}_K{self.K}/'
            else:
                folder = folder + f'/N{self.N}_K{self.K}/'
                for param_name in parameters:
                    value = self.metadata.get(param_name)
                    param_name = param_names(param_name)
                    folder = folder + f'/{param_name}{str(value)}/'
            path = os.path.join(script_dir, folder)
        if not os.path.isdir(path):
            os.makedirs(path)
        filepath = os.path.join(path, filenameMetadata)
        self.metadata.update({
            'experiment': self.experiment,
            'duration': self.duration,
            'nbTrials': self.nbTrials,
            'seed': self.seed})
        np.save(filepath, self.metadata)
        return self.metadata

    ###########################################################################
    # Utility
    ###########################################################################
    def reset(self, deep=False, seed='all'):
        '''
        Reset parameters of data, simulation and state, but not the network 
        itself, architecture and weights are kept.

        Parameters
        ----------
        deep : boolean 
            1. False : (default)
                All parameter used within a given simulation are reset.
            2. True :
                All parameter used within a given simulation are reset 
                +Parameters related to all simulations are reset, comprising 
                metadata.
        '''
        self.x = np.zeros((1, self.N))
        self.spikeCount = []
        self.activeIndex = []
        self.spikeDiagram = []
        # Verify if input should be triggered next trial
        if self.inputStream is not None:
            self.inputCond = True

        if deep:
            self.nbTrials = 0
            if inputLayer:
                self.u = np.zeros((self.inSize, self.N))
                self.inputStream = None
            if self.inputcurrent == True:
                self.c = 0
            self.duration = 0
            self.inputCond = False
            self.spikeCountTrials = []
            self.activeIndexTrials = []
            self.fileSpec = None
            self.metadata = {}            
            if seed == 'all':
                self.seed = {}
            elif seed == 'reservoir':
                self.seed.pop('seedWeights')


    def displayInfoNetwork(self):
        print('---- Network informations ----')
        print('Number of neurons N :', self.N)
        print('Connectivity degree K :', self.K)
        print('Mean weights :', self.meanWeights)
        print('Std weights :', self.stdWeights)

    def setSeedNetwork(self, seed=None, tag=''):
        """
        Intended to ease reproducibility, this method will
        be used by default by any of these methods :
            1. initRandomArchitecture()
            2. initGaussianWeight()
            3. initRandomState()

        Parameters
        ----------
        - seed : int or None
            1. None : (default)
                By default the seed will be generated with a uniform
                random function up to 10000.
            2. int :
                The seed will be set to the given integer value.
        - tag : str
            1. '' : (default)
                If no specific tag is given, the seed will be saved
                into a dictionary at the key : 'seed'.
            2. str :
                If a specific tag is given the seed will be saved
                into a dictionary at the key : 'seed'+tag.
        NB : each of the three mentioned methods send a different tag,
        best use is to put a different tag for each specific call of
        a method.
        """
        # Generate a random seed
        if seed is None: 
            seed = np.random.randint(100000)
            seeds = self.seed.get('seed' + tag, None)
            if seeds is None: 
                seed = np.random.randint(100000)
            else:
                # Generate unique seed
                if isinstance(seeds, list):  # Check if list
                    while seed in seeds:
                        seed = np.random.randint(100000)
                else:  # if not then it is a singleton
                    while seed == seeds:
                        seed = np.random.randint(100000)
                        
        np.random.seed(seed)  # needed for weights and state init
        rnd.seed(seed)  # needed for architecture init
        
        # Update metadata
        if self.seed.get('seed' + tag, False) != False:
            val_append(self.seed, 'seed' + tag, seed)
        else:
            self.seed.update({'seed' + tag: seed})