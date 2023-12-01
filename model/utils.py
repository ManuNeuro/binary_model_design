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
# Date: March 25th, 2021
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada

import numpy as np
import matplotlib.pyplot as plt
import glob
import math as mt
import os
import sys
import pandas as pd
import random as rnd
from scipy import sparse
import scipy.stats as stats
from typing import Union
from functools import wraps
from scipy.optimize import curve_fit
from scipy.stats import entropy
#from snoop import snoop
from numpy.fft import fft, ifft, fftshift
from pathlib import Path
from hurst import compute_Hc
from scipy.stats import norm, uniform
from scipy import special as sp

import os
PATH_ROOT = os.path.abspath(os.path.join('..'))


#############################################################
# %% Functions used inside the class
#############################################################
# https://devenum.com/append-value-to-an-existing-key-in-a-python-dictionary/
def val_append(dict_obj, key, value):
    if key in dict_obj:
        if not isinstance(dict_obj[key], list):
            # converting key to list type
            dict_obj[key] = [dict_obj[key]]
            # Append the key's value in list
            dict_obj[key].append(value)
        else:
            dict_obj[key].append(value)


def accurateDistribution(mean, std, nbSamples, option='normal'):
    if option == 'normal':
        var = (std) ** 2
        distribution = np.random.normal(mean, mt.sqrt(var), nbSamples)
    elif option == 'uniform':
        low = mean - std
        high = mean + std
        mean = (1/2) * (high + low)
        var = (1/12) * (high - low)**2
        distribution = np.random.uniform(low, high, nbSamples)
    if var != 0:
        correctedDist = distribution
        for i in range(50):
            # Correct mean
            realMean = np.mean(correctedDist)
            correctedDist = correctedDist + (mean - realMean)
    
            # Correct var
            epsilon = var / np.var(correctedDist)
            correctedDist = correctedDist * mt.sqrt(epsilon)
    else:
        correctedDist = distribution
    return correctedDist

#############################################################
# %% Function for random initialization
#############################################################
def generateRandomArchitecture(brain, option, layer, **kwargs): 
    # Keyword arguments
    I=kwargs.get('I', None)
    exclusion = kwargs.get('exclusion', None)
    seed = kwargs.get('seed', None )  
    # Check I bounds
    if I is not None:
        if I<0 and I>1:
            raise Exception(f'in "generateRandomArchitecture()", {layer} layer: I must be in [0, 1]')
    # Select layer
    if layer=='input':
        tag='archIn'
        size = brain.inSize
    elif layer=='readout':
        tag='archOut'
        size = brain.outSize
    else:
        raise Exception('generateRandomArchitecture only apply to input and readout layers.')
    # Set seed
    brain.setSeedNetwork(seed=seed, tag=tag)
    # Check option 
    idxNet = list(np.arange(0, brain.N))
    if option == 'global':
        # Reduce possible input to a subset of the network
        idxNet = rnd.sample(idxNet, int(I*brain.N))
        K = int(len(idxNet)/size)
    elif option == 'local':
        K = int(I*brain.N)
    elif option == 'exclusion' and exclusion is not None:
        newIdx = np.delete(idxNet, exclusion.astype(int))
        print('generate, idxNet', len(newIdx))
        idxNet = rnd.sample(list(newIdx), np.clip(int(I*brain.N), 0, len(newIdx)))
        K = int(len(idxNet)/size)
        print('generate, K', K)
    else:
        raise Exception('in "generateRandomArchitecture()" : option not supported.')
        
    # Generate architecture
    A = []
    for n in range(size):
        idxSelected = rnd.sample(list(idxNet), min([K, len(idxNet)]))
        idxNet = list(set(idxNet) - set(idxSelected))
        A.append(idxSelected)
    return A
    
def generateRandomWeights(brain, layer, distribution='gaussian', seed=None, **kwargs):
    # Select layer
    if layer=='input':
        tag='weightIn'
        size = brain.inSize
        A = brain.Ain
    elif layer=='readout':
        tag='weightOut'
        size = brain.outSize
        A = brain.Aout
    else:
        raise Exception('generateRandomArchitecture only apply to input and readout layers')
        
    # Set seed
    brain.setSeedNetwork(seed=seed, tag=tag)

    # Generate random weights
    if distribution == 'gaussian':
        mu = kwargs.get('mu', 0.0)
        sigma = kwargs.get('sigma', 0.5)
        weights = [np.random.normal(mu, sigma, len(A[i])) for i in range(size)]
    elif distribution == 'uniform':
        minW = kwargs.get('min', -0.5)
        maxW = kwargs.get('max', 0.5)
        weights = [np.random.uniform(minW, maxW, len(A[i])) for i in range(size)]

    return weights

def initRandomArchitecture(brain, K, seed=None):
    brain.setSeedNetwork(seed, tag='Architecture')
    # Initialize architecture
    idx = list(np.arange(0, brain.N))
    A = []
    for i in range(brain.N):
        A.append(rnd.sample(idx, K))  # Unique inputs
    brain.A = np.array(A)
    brain.K = K
    brain.nbSynapse = brain.K*brain.N

    # metadata
    brain.metadata.update({'K': K})

def initDistributionArchitecture(brain, K, distribution='poisson', seed=None):
    brain.K = K
    brain.setSeedNetwork(seed, tag='Architecture')
    indexes = np.arange(0, brain.N)
    rng = np.random.default_rng(seed)
    A = []
    nbSynapse = 0
    for j in range(brain.N):
        if distribution == 'poisson':
            K = rng.poisson(brain.K, 1); K = K[0]
        elif distribution == 'uniform':
            K = rnd.randint(1, brain.K)
        else:
            pass
        inputs = rnd.sample(list(indexes), K)
        A.append(inputs)
        nbSynapse += len(inputs)
    brain.A = A
    brain.nbSynapse = nbSynapse
    # metadata
    brain.metadata.update({'K': K})


def initGaussianWeights(brain, mu, sigma, seed=None):
    brain.setSeedNetwork(seed, tag='Weights')
    W = accurateDistribution(mu, sigma, brain.N * brain.K)
    brain.meanWeights = np.round(np.mean(W), 3)
    brain.stdWeights = np.round(np.std(W), 6)
    brain.W = W.reshape(brain.N, brain.K)
    brain.W_matrix = np.zeros((brain.N, brain.N))
    for i in range(brain.N):
        idxTarget = np.array(brain.A[i], int)
        brain.W_matrix[i, idxTarget] = brain.W[i]
    brain.W_matrix = sparse.csr_matrix(brain.W_matrix)  # Sparse representation speed-up by 30 times
    
    if abs(brain.meanWeights) < 10**(-5):
        brain.meanWeights = 0.0
        
    # metadata
    brain.metadata.update({'meanWeights': brain.meanWeights,
                          'stdWeights': brain.stdWeights})

def exp_2(x, mu=0, sigma=1):
    return norm.pdf(x, mu, sigma)

def bimodal(x, d=0.5, mus=[-0.5, 0.5], sigmas=[0.1, 0.1]):
    if d > 1:
        d = 1
    if d < 0:
        d = 0
    norm1 = exp_2(x, mus[0], sigmas[0])
    norm2 = exp_2(x, mus[1], sigmas[1])
    return (1-d)*norm1 + d*norm2

def generateRandomBimodal(nb_weights, **kwargs):
    mus = kwargs.get('mus', [-0.5, 0.5])
    sigmas = kwargs.get('sigmas', [0.1, 0.1])
    x = np.arange(-abs(mus[0])-10*sigmas[0], mus[1]+10*sigmas[1], 0.00001)
    pdf = bimodal(x, **kwargs)
    weights = np.random.choice(x, size=nb_weights, p=pdf/np.sum(pdf)) 
    return weights

def initBimodalWeights(brain, seed=None, **kwargs):
    brain.setSeedNetwork(seed, tag='Weights')
    W = generateRandomBimodal(brain.N * brain.K, **kwargs)
    brain.meanWeights = np.round(np.mean(W), 3)
    brain.stdWeights = np.round(np.std(W), 6)
    brain.d = kwargs.get('d')
    brain.W = W.reshape(brain.N, brain.K)
    brain.W_matrix = np.zeros((brain.N, brain.N))
    for i in range(brain.N):
        idxTarget = np.array(brain.A[i], int)
        brain.W_matrix[i, idxTarget] = brain.W[i]
    brain.W_matrix = sparse.csr_matrix(brain.W_matrix)  # Sparse representation speed-up by 30 times
    # metadata
    brain.metadata.update({'d':brain.d,
                           'meanWeights': brain.meanWeights,
                           'stdWeights': brain.stdWeights})
    

def initUniformWeights(brain, mu, sigma, seed=None):
    brain.setSeedNetwork(seed, tag='Weights')
    W = accurateDistribution(mu, sigma, brain.N * brain.K, option='uniform')
    brain.meanWeights = np.round(np.mean(W), 3)
    brain.stdWeights = np.round(np.std(W), 6)
    brain.W = W.reshape(brain.N, brain.K)
    brain.W_matrix = np.zeros((brain.N, brain.N))
    for i in range(brain.N):
        idxTarget = np.array(brain.A[i], int)
        brain.W_matrix[i, idxTarget] = brain.W[i]
    brain.W_matrix = sparse.csr_matrix(brain.W_matrix)  # Sparse representation speed-up by 30 times
    # metadata
    brain.metadata.update({'meanWeights': brain.meanWeights,
                          'stdWeights': brain.stdWeights})


def initDistributionWeight(brain, seed=None, **kwargs):
    if brain.A is None:
        raise Exception('in "initDistributionWeight()" : you need to initialize architecture before weights')
    brain.setSeedNetwork(seed, tag='Weights')
    distribution = kwargs['distribution']
    mu = kwargs['meanWeight']
    sigma = kwargs['stdWeight']
    if distribution == 'gaussian':
        weightsDist = accurateDistribution(mu, sigma, brain.nbSynapse)
        # metadata
    elif distribution == 'asymetric':
        # Generate gaussian probability distribution 
        ratio = kwargs['ratio']
        Pos = ratio/(1+ratio)
        Neg = 1-Pos
        normal_dist = stats.norm(loc=mu, scale=sigma)
        delta = 1e-6
        x = np.arange(-abs(mu)-5*sigma, +abs(mu)+5*sigma, delta)
        pmfNormal = normal_dist.pdf(x)*delta
        # Pmf truncated of positive and negative values
        pmfPos = np.array(pmfNormal)
        pmfNeg = np.array(pmfNormal)
        idxNeg = np.where(x<0); idxNeg=idxNeg[0]
        idxPos = np.where(x>=0); idxPos=idxPos[0]
        pmfPos = pmfPos[idxPos]
        pmfNeg = pmfNeg[idxNeg]
        # Final concateneated distribution 
        pmfPos = pmfPos/sum(pmfPos)*Pos
        pmfNeg = pmfNeg/sum(pmfNeg)*Neg
        # Generate weights
        asymetricGaussianDistribution = np.hstack((pmfNeg, pmfPos))
        weightsDist = rnd.choices(x, asymetricGaussianDistribution, k=brain.nbSynapse)
        # metadata
        brain.metadata.update({'ratio':ratio})
    elif distribution == 'uniform':
        # Generate asymetric uniform probability distribution 
        ratio = kwargs['ratio']
        Pos = ratio/(1+ratio)
        Neg = 1-Pos
        mu =-0.1
        sigma = 0.3
        scale = 2*sigma
        loc = mu-sigma
        uniform_dist = stats.uniform(loc=loc, scale=scale)
        delta = 1e-6
        x = np.arange(-abs(mu)-1.1*sigma, +abs(mu)+1.1*sigma, delta)
        pmfUniform = uniform_dist.pdf(x)*delta
        # Pmf truncated of positive and negative values
        pmfPos = np.array(pmfUniform)
        pmfNeg = np.array(pmfUniform)
        idxNeg = np.where(x<0); idxNeg=idxNeg[0]
        idxPos = np.where(x>=0); idxPos=idxPos[0]
        pmfPos = pmfPos[idxPos]
        pmfNeg = pmfNeg[idxNeg]
        # Final concateneated distribution 
        pmfPos = pmfPos/sum(pmfPos)*Pos
        pmfNeg = pmfNeg/sum(pmfNeg)*Neg
        # Generate weights
        asymetricUniformDistribution = np.hstack((pmfNeg, pmfPos))
        weightsDist = rnd.choices(x, asymetricUniformDistribution, k=brain.nbSynapse)
        # metadata
        brain.metadata.update({'ratio':ratio})
        
    # Assign the weights
    W = []
    idxCount = 0
    for n in range(brain.N):
        K = len(brain.A[n])
        weights = list(weightsDist[idxCount:idxCount+K])
        W.append(weights)
        idxCount += K
    brain.setWeights(W)                
    # metadata
    brain.metadata.update({'meanWeightsOrig': mu,
                           'stdWeightsOrig': sigma})

#############################################################
# %% Functions for dimension checking
#############################################################

def checkStateDim(x, N):
    if np.shape(x) != (1, N):
        raise Exception(f'State dimension do not match [1, {N}]')

# https://stackoverflow.com/questions/47492204/check-if-matrix-is-square
def is_squared(matrix):
   # Check that all rows have the correct length, not just the first one
   return all(len(row) == len(matrix) for row in matrix)


def sizeList(list_: list) -> int:
    '''
    Return the number of single element in list
    '''
    sizes = []
    for element in list_:
        sizes.append(np.array(element).size)
    return sizes

def dimSwapCheckIn(data: np.ndarray, Nin: int) -> np.ndarray:
    '''
    Check if one of the dimensions 
    of data matches Nin :
    - If the first one does, data is returned.
    - If the second one does, data is transposed.
    - Otherwise there is an error.
    '''
    count = 0
    dimensions = np.shape(data)
    while (dimensions[0] != Nin):
        # Condition for breaking the loop
        if count == 1:
            count += 1
            break
        # Transpose
        dimensions = np.shape(data.T)
        count += 1
    if count == 0:
        return data
    elif count == 1:
        return data.T
    elif count == 2:
        raise Exception(f'No dimension of data match Nin, {np.shape(data)}')

    
def checkDim(data: Union[list, np.ndarray], **kwargs) -> (list, str):
    '''
    Take list or array; one or multi-dimensional.
    Check dimension fitting with given parameters
    Format into a one dimensional list
    of integer, or array, with Nin dimension.
    '''
    # Get kwargs
    try:
        Nin = kwargs['Nin'] 
        Nout = kwargs['Nout']
    except Exception as error:
        print(error)
    # Check dimension and format
    if isinstance(data, list):
        # 2D with multidimensionaly input list
        sizes = sizeList(data)
        if int(sum(sizes)/max(sizes)) != len(sizes):
            flag = 'multiSize'
            if len(data) != Nin:
                raise Exception(f'Dimension in {len(data)} do not coincide with given parameter {Nin}')
            for n in range(Nin):   
                # Check that all projecting inputs fit the size of network reservoir
                if len(np.array(data[n], dtype=object)) > Nout: 
                    raise Exception('Number of input projection is higher than out dimension')
            return [np.array(data[n]) for n in range(Nin)], flag
    if isinstance(data, list) or isinstance(data, np.ndarray):
        flag = 'uniqueSize'
        dimensions = np.shape(data)
        # 1D array, of multidimensional list
        if len(dimensions) == 1:
            if len(data) != Nin and Nin == 1: 
                if len(data) > Nout:
                    raise Exception(f'Dimension out {len(data)} is higher than given parameter {Nout}')
                data = [np.array(data)]
            elif len(data) != Nin and Nin != 1:
                raise Exception(f'Dimension in {len(data)} do not coincide with given parameter {Nin}')
            else:
                data = list(data)
        elif len(dimensions) == 2:
            # Check in dimension and swap if needed
            data = np.array(data)
            data = dimSwapCheckIn(data, Nin)
            dimensions = np.shape(data)
            if dimensions[1] > Nout:
                raise Exception('Dimension out is higher than given parameter')
            # Transform into list
            data = [data[n, :] for n in range(dimensions[0])]
        return data, flag

#############################################################
# %% Functions for initializing circuit
#############################################################

def formatDimension(func):
    '''
    Check dimensions of all non keyword argument
    And format them to list of arrays.
    Then call the function with formated argument
    '''
    @wraps(func)
    def wrapper(*arg, **kwargs):
        newArg = []
        print(f"Format dimension for: {func.__name__}")
        for i, data in enumerate(arg):
            dataFormated, flag = checkDim(data, **kwargs)
            print(f"Dimension after cheking: {type(dataFormated)}, {np.shape(dataFormated)}")
            newArg.append(dataFormated)
            kwargs.update({'flag':flag})
        return func(*newArg, **kwargs)
    return wrapper


@formatDimension
def instantiateArchitecture(A: Union[list, np.ndarray], **kwargs) -> list:
    '''
    Check dimensions with formatDimension, 
    Then return architecture appropriately formated in a list of array.

    Parameters
    ----------
    # Decorators argument #
    (/!\ Prevail over function argument)
    Nin : int
        Input dimension
    Nout : int
        Output dimension
    architecture :list or array 
        Architecture of the input layer.
        Accepted : 1D, 2D, and 2D with multiple dimension sizes.
    # Function argument #
    flag : str
        The output of format dimensions, taken as 
        input from setArchitectureIn.
        This flag indicate weather the dimensions are multisizes 
        or identicaly sized.
    '''
    try:
        flag = kwargs['flag']
    except:
        print('in instantiateArchitecture() : "flag" key is missing in kwargs dictionnary')
    if flag == 'uniqueSize':
        A = np.array(A)
        #A = np.sort(A, axis=1)
        K = np.shape(A)[1]
    elif flag == 'multiSize':
        totalK = 0
        for i in range(len(A)):
            #A[i] = np.sort(A[i])
            totalK += len(A[i])
        A = np.array(A, dtype=object)
        K = np.round(totalK/len(A), 1) # Average K
    else:
        raise Exception('flag value not supported')
        
    return list(A), K


@formatDimension
def instantiateWeights(W: Union[list, np.ndarray], 
                       **kwargs):
    '''
    Check dimensions with formatDimension, 
    Then instantiate Weights in network and return an array, 
    formated list of non zero weights, and distribution statistics.

    Parameters
    ----------
    # Decorators argument #
    (/!\ Prevail over function argument)
    Nin : int
        Input dimension
    Nout : int
        Output dimension
    Weights :list or array 
        Weights of the input layer.
        Accepted : 1D, 2D, and 2D with multiple dimension sizes.
    # Function argument #
    flag : str
        The output of format dimensions, taken as 
        input from setArchitectureIn.
        This flag indicate weather the dimensions are multisizes 
        or identicaly sized.
    '''
    # Arguments
    try:
        flag = kwargs['flag']
        Nin = kwargs['Nin']
        Nout = kwargs['Nout']
        A = kwargs['A']
    except Exception as error:
        print(error)
    option = kwargs.get('option', 'array')
    
    # Compute statistics
    if flag == 'uniqueSize':
        buffW = W
    elif flag == 'multiSize':
        buffW = np.hstack(W)
    else:
        raise Exception('flag value not supported')
    meanWeights = np.round(np.mean(buffW), 4)
    stdWeights = np.round(np.std(buffW), 6)
    
    # Create a matrix
    W_matrix = np.zeros((Nin, Nout))
    for i in range(Nin):
        idxTarget = np.array(A[i], int)
        if len(W_matrix[i, idxTarget]) != len(W[i]):
            raise Exception('In setWeight : weights and architecture dimensions must agree')
        W_matrix[i, idxTarget] = W[i]
    
    # Option of output type
    if option == 'array':
        pass
    elif option == 'sparse':
        W_matrix =  sparse.csc_matrix(W_matrix)
    
    if abs(meanWeights) < 10**(-5):
        meanWeights = 0.0
    
    return W_matrix, W, (meanWeights, stdWeights)



#############################################################
# %% Functions for inputs
#############################################################
@formatDimension
def instantiateInputs(inputs: Union[list, np.ndarray],
                      timings: Union[list, np.ndarray], 
                      **kwargs):
    '''
    Check dimensions with formatDimension, 
    Then return inputs appropriately formated as array or sparse matrix.

    Parameters
    ----------
    # Decorators argument #
    (/!\ Prevail over function argument)
    Nin : int
        Input dimension
    Nout : int
        length of the input stimulus
    architecture :list or array 
        Architecture of the input layer.
        Accepted : 1D, 2D, and 2D with multiple dimension sizes.
    # Function argument #
    flag : str
        The output of format dimensions, taken as 
        input from setArchitectureIn.
        This flag indicate weather the dimensions are multisizes 
        or identicaly sized.
    option : str
        'array' : return an array
        'sparse' : uses scipy.sparse to convert array and return
        a sparse matrix.
    '''
    # Arguments
    try:
        flag = kwargs['flag']
        Nin = kwargs['Nin']
    except Exception as error:
        print(error)
    option = kwargs.get('option', 'array')
    
    # Get time size
    if flag == 'uniqueSize':
        timings = np.array(timings)
        Tend = np.amax(timings)
        T0 = np.amin(timings)
    elif flag == 'multiSize':
        timings = np.array(timings, dtype=object)
        #print(type(timings), np.shape(timings))
        Tend = max([max(timings[i]) for i in range(len(inputs))])
        T0 = min([min(timings[i]) for i in range(len(inputs))])
    else:
        raise Exception('flag value not supported')
    
    if len(np.shape(inputs)) != 2:
        # Create input matrix
        u = np.zeros((Nin, Tend+1))
        for i in range(Nin):
            idxTimings = np.array(timings[i], int)
            if len(u[i, idxTimings]) != len(inputs[i]):
                raise Exception('In instantiateInputs : input and architecture dimensions must agree')
            u[i, idxTimings] = inputs[i]
    else:
        u = inputs
        
    times = (T0, Tend)
    # Opption of output type
    if option == 'array':
        pass
    elif option == 'sparse':
        u =  sparse.csc_matrix(u)
    u = u.transpose()
    return u, times

# Outdated, kept for compatibility with older code
def randomInputs_ND(brain, u, timings, I, seed=None, tag=None):
    idxList = list(np.arange(0, brain.N))
    
    # Check I bounds
    if I<0 and I>1:
        raise Exception('in "randomInputs_ND() : I must be in [0, 1]')
    
    if len(u) == 1 and len(timings)>1:
        u = u*np.ones(len(timings))
        
    if seed is None:
        seed = np.random.randint(10000)
    rnd.seed(seed)
    
    # Generate random input
    inputs = []
    idxSelected = rnd.sample(idxList, int(I*brain.N))
    for j in range(len(timings)):
        vector = np.zeros(brain.N)
        vector[idxSelected] = 1*u[j]
        sparseVector = sparse.csr_matrix(vector)
        inputs.append(sparseVector.T) 
        # print(len(sparseVector.indices))
    brain.setInputs(inputs, timings, tag)
    
    # Update metadata
    if brain.seed.get('seedInput', False) != False:
        val_append(brain.seed, 'seedInput', seed)
    else:
        brain.seed.update({'seedInput': seed})
    return inputs
        
#############################################################
# %% Functions fot plotting data
#############################################################
def plotData(brain, path=None, addPath='', tag=''):
    # Filename
    plt.ioff()
    if brain.fileSpec is None:
        brain.fileSpec = 'N{0}_K{1}_D{2}_T{3}_W{4}_std{5}'.format(brain.N, brain.K, brain.duration, brain.nbTrials,
                                                                  brain.meanWeights, brain.stdWeights)
    if tag != '':
        tag = tag + '_'
    # Path
    if path is None:
        script_dir = os.path.dirname(__file__) + '/'
        script_dir = splitpath(path, 'model')
        script_dir = script_dir + '/results/'+brain.sim
        folder = os.path.join(brain.experiment, addPath)
        folder = folder +'/N{0}_K{1}/mu{2}/std{3}'.format(brain.N, brain.K, brain.meanWeights,
                                                                     brain.stdWeights)
        path = os.path.join(script_dir, folder)
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Save plot in :', path)
    if brain.spikeCountTrials != []:
        for t in range(brain.nbTrials):
            spikeCount = brain.spikeCountTrials[t]
            filenameImage = 'spikeCount_' + str(t) + '_' + tag + brain.fileSpec + '.png'
            pathSC = os.path.join(path, 'spikeCount/')
            if not os.path.isdir(pathSC):
                os.makedirs(pathSC)
            pathImage = pathSC + filenameImage
            # print(pathImage)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('time step t')
            ax.set_ylabel('A(t)')
            ax.set_title(
                'Spike count over time, $\mu(W)$={0}, $\sigma(W)$={1}'.format(brain.meanWeights, brain.stdWeights))
            fig.suptitle('N={0}, K={1}, duration={2}'.format(brain.N, brain.K, brain.duration))
            ax.plot(spikeCount, label='T:{0}'.format(t))
            ax.legend()
            fig.savefig(pathImage)
            plt.close()

    if brain.spikeDiagram != []:
        pass  # ToDo

def plotWeightDistribution(brain, path=None, addPath='', tag=''):
    # Filename
    if brain.fileSpec is None:
        brain.fileSpec = 'N{0}_K{1}_D{2}_T{3}_W{4}_std{5}'.format(brain.N, brain.K, brain.duration, brain.nbTrials,
                                                                  brain.meanWeights, brain.stdWeights)
    if tag != '':
        tag = tag + '_'
    # Path
    if path is None:
        script_dir = os.path.dirname(__file__) + '/'
        script_dir = splitpath(path, 'model')
        script_dir = script_dir + '/results/'+brain.sim
        folder = os.path.join(brain.experiment, addPath)
        folder = folder +'/N{0}_K{1}/mu{2}/std{3}'.format(brain.N, brain.K, brain.meanWeights,
                                                                     brain.stdWeights)
        path = os.path.join(script_dir, folder)
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Save plot in :', path)
        
    weights = brain.W     
    N = brain.N
    K = brain.K
    
    # Curve fitting Normal distribution
    meanWeight = brain.metadata['meanWeightsOrig']
    stdWeight = brain.metadata['stdWeightsOrig']
    print('mu(W)=', meanWeight)
    print('std(W)=', stdWeight)
    normal_dist = stats.norm(loc=meanWeight, scale=stdWeight)
    
    # Weight distribution
    weightSequence = []
    for i in range(N):
        if i == 0:
            weightSequence = weights[i]
        else:
            weightSequence = np.hstack((weightSequence, weights[i]))
    
    nbBins, weights_array = np.histogram(weightSequence, bins=200)
    pW = nbBins/sum(nbBins) 
    
    
    delta = 1e-4
    # need very large x to compute correctly
    x = weights_array[1:]
    delta = abs(x[4]-x[5])
    idxNeg = np.where(x<0); idxNeg=idxNeg[0]
    idxPos = np.where(x>=0); idxPos=idxPos[0]
    
    pmfNormal = normal_dist.pdf(x)*delta
    pmfPos = np.array(pmfNormal)
    pmfNeg = np.array(pmfNormal)
    pmfPos[idxNeg]=0
    pmfNeg[idxPos]=0
    
    pmfPos = pmfPos/sum(pmfPos)
    pmfNeg = pmfNeg/sum(pmfNeg)
    
            
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    ratio = np.round(sum(pW[idxPos])/sum(pW[idxNeg]), 4) 
    ax.set_title(f'Weight distribution, fitted with two truncated gaussians. R={ratio}')
    ax.set_ylabel('P(W)')
    ax.set_xlabel('W')
    plt.plot(x, pW, '.-', label='weight distribution')
    ax.plot(x, pmfPos*sum(pW[idxPos]), '--', label='Pos_gauss*P(W>0):{0}'.format(np.round(sum(pW[idxPos]), 3)))
    ax.plot(x, pmfNeg*sum(pW[idxNeg]), '--', label='Neg_gauss*P(W<0):{0}'.format(np.round(sum(pW[idxNeg]), 3)))
    ax.legend(title='$\mu={0}$, $\sigma(W)={1}$'.format(meanWeight, stdWeight))
    fig.savefig(path+'/weightDistribution_R{}N{}_K{}_W{}_std{}.png'.format(ratio, N, K, meanWeight, stdWeight))
    plt.close()
    print('Proportion of positive synapse:', sum(pW[idxPos]))
    print('Proportion of Negative synapse:', sum(pW[idxNeg]))
    print('Ratio of excitatory and inhibitory synapses:', sum(pW[idxPos])/sum(pW[idxNeg]))
    print('Total mass of positive synapse:', sum(nbBins[idxPos]*x[idxPos]))
    print('Total mass of negative synapse:', sum(nbBins[idxNeg]*abs(x[idxNeg])))
    print('average of the pruned weight distribution :', np.mean(weightSequence))
    print('Standard deviation of the pruned weight distribution :', np.std(weightSequence))

        
#############################################################
# %% Functions fot saving datas
#############################################################
def dataPath(sim, experiment, N=None, K=None, meanWeights=None, stdWeights=None,
             I=None, u=None, nbImpulse=None, **kwargs):
    # Get directory of this script
    folderResult = 'results/' + sim + '/' + experiment
    if nbImpulse is not None:
        folderNetwork = '/nbImpulse'+str(nbImpulse)
    else:
        folderNetwork = ''
    if I is not None:
        if u is None:
            folderNetwork = folderNetwork + '/noInput_I'+str(I)
        else:
            folderNetwork = folderNetwork + '/input'+str(u)+'_I'+str(I)

    if N is not None and K is not None:
        folderNetwork = folderNetwork + '/N{0}_K{1}'.format(N, K)
        if meanWeights is not None:
            folderNetwork = folderNetwork + '/mu{0}/'.format(meanWeights)
            if stdWeights is not None:
                folderNetwork = folderNetwork + '/std{0}'.format(stdWeights)
        for k, v in kwargs.items():
            folderNetwork = folderNetwork + f'/{k}' + str(v) + '/'

    results_dir = os.path.join(PATH_ROOT, folderResult + folderNetwork)
    #print(results_dir)
    return results_dir


def dataPathPerf(sim, experiment, task=None, optimizer=None, activation=None, nbtrial=None, length=None, delay=None, **kwargs):
    # Get directory of this script

    print('SCRIPT:', PATH_ROOT)
    folderResult = 'results/' + sim + '/' + experiment
    if task is not None:
        folderNetwork = '/' + task
    else:
        folderNetwork = ''
    if optimizer is not None:
        if activation is not None:
            folderNetwork = folderNetwork + '/' + optimizer + '_' + activation
        else:
            folderNetwork = folderNetwork + '/' + optimizer
    if nbtrial is not None:
        folderNetwork = folderNetwork + '/nbtrial' + str(nbtrial)
    if length is not None:
        folderNetwork = folderNetwork + '/length' + str(length) + '/'
    if delay is not None:
        folderNetwork = folderNetwork + '/delay' + str(delay) + '/'
    for k, v in kwargs.items():
        folderNetwork = folderNetwork + f'/{k}' + str(v) + '/'

    
    results_dir = os.path.join(PATH_ROOT, folderResult + folderNetwork)
    return results_dir

def dataPathNew(sim, experiment, N=None, K=None, meanWeights=None, stdWeights=None,
             I=None, u=None, nbImpulse=None, ratio=None):
    # Get directory of this script
    script_dir = os.path.dirname(__file__) + '/'
    script_dir = script_dir.split('model/')[0]
    print('SCRIPT', script_dir)
    folderResult = 'results/' + sim + '/' + experiment
    if nbImpulse is not None:
        folderNetwork = '/nbImpulse'+str(nbImpulse)
    else:
        folderNetwork = ''
    if I is not None:
        if u is None:
            folderNetwork = folderNetwork + '/noInput_I'+str(I)
        else:
            folderNetwork = folderNetwork + '/input'+str(u)+'_I'+str(I)

    if N is not None and K is not None:
        folderNetwork = folderNetwork + '/N{0}_K{1}'.format(N, K)
        if meanWeights is not None and stdWeights is not None:
            folderNetwork = folderNetwork + '/mu{0}_std{1}/'.format(meanWeights, stdWeights)
            if ratio is not None:
                folderNetwork = folderNetwork + 'R{0}/'.format(ratio)

    results_dir = os.path.join(script_dir, folderResult + folderNetwork)
    #print(results_dir)
    return results_dir

#############################################################
# %% Functions for editing files
#############################################################

# Function to rename multiple data filename
def renameFile(key, N, K, meanWeights, sim, experiment, I=None):
    directory = dataPath(sim, experiment, N=N, K=K, meanWeights=meanWeights, I=I)
    filenames = glob.glob(directory+'/metadata*.npy')
    for filepath in filenames:
        metadata = np.load(filepath, allow_pickle='TRUE').item()
        path = metadata['dataPath']
        seeds = metadata['seed']
        seed = seeds['seedWeights']
        print('------')    
        # str1, str2 = directory.split(experiment)
        # directory = str1+sim+'/'+experiment+str2
        cut = filepath.split('_N')[0]
        split = filepath.split(cut)[1]
        filename = f'metadata_seed{seed}'+split
        dst = directory+ '/' +filename
        src = filepath
        # rename() function will
        # rename all the files
        print('Old file name:')
        print(src)
        print('New filename:')
        print(dst)
        os.rename(src, dst)

def editMetadataPath(N, K, meanWeights, sim, experiment, I=None):
    directoryMetadata = dataPath(sim, experiment, N=N, K=K, meanWeights=meanWeights, I=I, inputs=False)
    print(directoryMetadata)
    filenames = glob.glob(directoryMetadata+'metadata*.npy')
    print('------')
    for filename in filenames:
        print('metadata :', filename)
        metadata = np.load(filename, allow_pickle='TRUE').item()
        directory = metadata['dataPath']
        print('------')    
        print('dataPath', directory)
        print(experiment)
        str1, str2 = directory.split(experiment)
        directory = str1+sim+'/'+experiment+str2
        print('------')
        print('new Directory :')
        print(directory)
        metadata['dataPath'] = directory
        print(' ---- Save new metadata -----')
        _, filename = filename.split('mu{0}'.format(meanWeights))
        print('filename :', filename)
        print(directoryMetadata+filename)
        np.save(directoryMetadata+filename, metadata)

def editMetadataItem(N, K, meanWeights, sim, experiment, I=None):
    item = 'idxActiveFile'
    directoryMetadata = dataPath(sim, experiment, N=N, K=K, meanWeights=meanWeights, I=I)
    print(directoryMetadata)
    filenames = glob.glob(directoryMetadata+'metadata*.npy')
    for filename in filenames:
        print('metadata :', filename)
        metadata = np.load(filename, allow_pickle='TRUE').item()
        directory = metadata['dataPath']
        print('------')
        print('dataPath', directory)
        print(metadata[item])
        
        # split = filename.split('metadata')[1]
        # itemFile = 'idxActive'+split.split('.npy')[0]
        # metadata[item] = itemFile
        # print(' ---- Save new item in metadata -----')
        # print('New item :', itemFile)
        # print(filename)
        # np.save(filename, metadata)
        # print('*****************')

def editMetadataIdx(N, K, meanWeights, sim, experiment, I=None):
    directoryMetadata = dataPath(sim, experiment, N=N, K=K, meanWeights=meanWeights, I=I, inputs=False)
    print(directoryMetadata)
    filenames = glob.glob(directoryMetadata+'metadata*.npy')
    print('------')
    labels = ['ChaoticAttractor', 'CyclicAttractor', 'DeadAttractor', 'FixAttractor']
    for filename in filenames:
        print('metadata :', filename)
        metadata = np.load(filename, allow_pickle='TRUE').item()
        print(' ---- Save new metadata -----')
        _, filename = filename.split('mu{0}'.format(meanWeights))
        for label in labels:
            if label in filename:
                _, splited = filename.split(label)
                break
        print('splited', splited)
        idx = splited.split('_')[0]
        metadata.update({'idx':idx,
                         'attractor':label})
        print(' ---- Save new metadata -----')
        print('filename :', filename)
        print(directoryMetadata+filename)
        np.save(directoryMetadata+filename, metadata)


#############################################################
# %% Functions for finding in list
#############################################################
def findIdx_notInList(list_, value):
    return [i for i, x in enumerate(list_) if x != value]


# Find sub-string in string
def findall(sub, string):
    return [n for n in range(len(string)) if string.find(sub, n) == n]


def splitpath(path, seperator):
    parts = Path(path).parts
    
    newPath = ''
    for part in parts:
        if part != seperator:
            newPath += part + '\\'
        elif part == seperator:
            break
    
    return newPath

#############################################################
# %% Functions for analysis
#############################################################

def get_hurst_exponent(time_series, max_lag=None, min_lag=20, plot=False,
                       rescale=False):
    """Returns the Hurst Exponent of the time series"""
    
    def linear(x, a, b):
        return a*x+b
    
    if not rescale:
        if max_lag is None:
            max_lag = len(time_series)-1

        lags = range(2, max_lag)
        # variances of the lagged differences        
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    
        # calculate the slope of the log plot -> the Hurst Exponent
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        H = reg[0]*2
        c = reg[1]
        val = [lags, tau]
        if plot:
            print(tau)
            plt.figure()
            plt.plot(lags, tau)
            # plt.plot(np.log(lags), linear(lags, reg[0], reg[1]),  '--', color='green')
            plt.title(f'H={H}')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
    else:
        H, c, val = compute_Hc(time_series, kind='change')
        # Plot the graph
        if plot:
            plt.figure()
            plt.plot(val[0], c*val[0]**H, color="blue")
            plt.scatter(val[0], val[1], color="red")
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Time interval')
            plt.ylabel('R/S ratio')
            plt.title(f'H={H}')
            plt.legend()
    return H, c, val 

def statisticsActivity(N, A_trials, startIdx, variance='activity'):
    # Put activity in a matrix        
    for i in range(len(A_trials)):
        A_trial = A_trials[i]
        A_trial = A_trial[startIdx:]
        if i == 0:
            A_allData = A_trial
        else:
            A_allData = np.vstack((A_allData, A_trial))
            # plt.figure()
            # plt.plot(A_trial)
            # plt.legend(title='idx:'+str(idx[i])+', var'+str(np.var(A_trial)))

    shape = np.shape(A_allData)
    if len(shape) > 1:
        # Average over time of each trials
        avgTime = np.mean(A_allData, axis=1)
        # Variance over time of each trials
        varTime = np.var(A_allData * N, axis=1) / N
    else:
        # Average over time of each trials
        avgTime = np.mean(A_allData)
        # Variance over time of each trials
        varTime = np.var(A_allData * N) / N
    avgTrial = np.mean(avgTime)
    if variance == 'reservoir':
        varTrial = np.var(avgTime)
    if variance == 'activity':
        varTrial = np.mean(varTime)

    return avgTrial, varTrial

def linearFit(x, a, b):
    return a*x+b

def expFitSimple(x, b):
    return np.exp(-x / b)

def fast_cross_correlation(x, y, norm=True):
    # https://lexfridman.com/fast-cross-correlation-and-time-series-synchronization-in-python/
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)
    
def my_corrcoef( x, y ):    
    mean_x = np.mean( x )
    mean_y = np.mean( y )
    std_x  = np.std ( x )
    std_y  = np.std ( y )
    n      = len    ( x )
    return (( x - mean_x ) * ( y - mean_y )).sum() / n / ( std_x * std_y )

def fastCrossCorr(x, maxlag, norm=True, plot=False):
    '''
    Compute the auto correlation of the signal x, with given 
    maximum time lag, goes from -maxlag to maxlag

    Parameters
    ----------
    x : array of float
        data signal, one dimensional.
    maxlag : positive int
        maximum lag used to compute cross-correlation.

    Returns
    -------
    R : array of float
        crosscorrelation vector
    lag : array of int
        lag vector
    '''
    x = x - np.mean(x)  # Assume 0 mean.

    T = len(x)
    if maxlag > T - 1:
        raise Exception('maxLag cannnot be smaller than len(x)-1')
    elif maxlag == 0:
        raise Exception('maxLag cannot be 0')

    # Zero padding, control max lag
    zeropPad = len(x) - 1
    x_zero = np.zeros(zeropPad + len(x))
    x_zero[:len(x)] = x
    # Compute fft (real signal)
    fftx = np.fft.rfft(x_zero)  # real fft (faster)
    # Compute correlation with fast fourrier transform
    corr = np.fft.irfft(fftx * np.conj(fftx))
    # cut signal in half and shift it to get it centered
    R = np.hstack((corr[len(corr) - maxlag:], corr[:maxlag + 1]))  # plus cut signal for appropriate time lag
    lag = np.arange(-maxlag, maxlag + 1)
    if norm:
        normVec = np.hstack(
            (np.arange(T - maxlag, T, 1), np.arange(T, T - maxlag - 1, -1)))  # to counter the effect of diminishing
        R = R / normVec

    if plot:
        plt.figure()
        plt.vlines(lag, R, ymax=0)
        plt.plot(lag, np.mean(x) * np.ones(len(lag)))
        plt.xlabel(r'$\delta t$ : time lag')
        plt.ylabel('C($\delta t$)')
        plt.title('Correlations, versus time lag')
        plt.show()

    return np.array(R), np.array(lag)


        
def tauEvoked(A, plot=False):
    A = np.array(A)
    y_data = np.array(A)
    x_data = np.arange(0, len(A))    
    y_data = y_data - np.mean(y_data[int(len(y_data)/2):len(y_data)])
    y_data = y_data/y_data[0]
        
    if plot:
        plt.figure()
        plt.plot(x_data, y_data)
    # Computing branching factor and time constant
    try:
        popt, pcov = curve_fit(expFitSimple, x_data, y_data)
        tau = popt[0]
        if plot:
            plt.plot(expFitSimple(x_data, tau), '--', color='red')
            plt.text(1, 1, f'tau = {popt[0]}')
        return tau, popt[0]
    except:
        print('Error 1')
        return None, None

def timeConstant(A, **kwargs):
    norm = kwargs.get('norm', True)
    maxlag = kwargs.get('maxlag', 800)
    plot = kwargs.get('plot', False)
    # Compute time constant of autocorrelation
    autoCorrA, lagVector = fastCrossCorr(A, norm=norm, maxlag=maxlag)
    idxZero = np.where(lagVector == 0);
    idxZero = idxZero[0]
    autoCorrA = autoCorrA / autoCorrA[idxZero]
    x_data = lagVector[lagVector >= 0]
    idxPos = np.where(lagVector >= 0)
    y_data = autoCorrA[idxPos]
    
    if plot:
        plt.figure(figsize=(15, 10))
        plt.plot(x_data, y_data, label='autocorrelation')
        plt.xlabel('$\delta$')
        plt.ylabel('C(A)')
    # Computing branching factor and time constant
    try:
        popt, pcov = curve_fit(expFitSimple, x_data, y_data)
        tau = popt[0]
        if plot:
            plt.plot(x_data, expFitSimple(x_data, tau), '--', color='red', label='Exponential fitting')
            plt.legend()
        return tau
    except:
        return None

def pruneNetwork(A, W, indexActives, test=False):
    architecture = np.array(A, dtype=object)
    weights = np.array(W, dtype=object)
    
    oldIdx = indexActives.astype(int)
    archGCC = architecture[oldIdx].copy()
    weightGCC = weights[oldIdx].copy() 
    
    # Create mapping of old->new indexes
    N1 = len(indexActives)
    newIdx = np.arange(N1)
    convertIdx = {}
    for i, j in zip(oldIdx, newIdx):
        convertIdx.update({int(i):j})
        
    # Remove pruned neurons
    newArchitecture = []
    newWeights = []
    print(N1)
    for i in range(N1):
        # Architecture
        intersectionIdx = list(set(archGCC[i]) & set(oldIdx)) # Keep the active index only
        newArchitecture.append(np.array(intersectionIdx))
        # Weights
        weights_i = weightGCC[i]
        newWeights_i = []
        for idx in intersectionIdx:
            idxKeep = np.where(np.array(archGCC[i]) == idx); idxKeep = idxKeep[0]
            newWeights_i.append(weights_i[idxKeep][0])
        newWeights.append(np.array(newWeights_i))
      
    # Change index
    for i, arch_i in enumerate(newArchitecture):
        for j, old in enumerate(arch_i):
            newArchitecture[i][j] = convertIdx[old]
    
    if not test:
        return newArchitecture, newWeights
    if test:
        return newArchitecture, newWeights, convertIdx

def test_attractor(A_trial, N):
    condDead = False
    condConst = False
    condCyclic = False
    condSaturated = False
    label = None
    # Test for dead and saturated case
    if np.mean(A_trial) == 0:
        label = "No activity"
        condDead = True
    if int(np.mean(A_trial)) == 1:
        label = "Saturated attractor"
        condSaturated = True
    # Test for constant activity
    if not condDead and not condSaturated:
        diffA = np.diff(A_trial, n=1)
        if diffA[diffA <= 1 / N].size == len(diffA):  # Compute first drivative
            label = "Fixed point attractor"
            condConst = True
    # Test for cyclic attractor
    if condConst == condDead == condSaturated == False:
        chunk_length = int(len(A_trial) * 0.1)
        chunk_A = A_trial[-chunk_length:]
        for i in range(0, len(A_trial)-2*chunk_length, 1):
            window_A = A_trial[i:i+chunk_length]
            test_period = chunk_A - window_A
            if test_period[test_period!=0].size ==0:
                label = "Cyclic attractor"
                condCyclic = True
                break
    # Last possibility is chaotic
    if condConst == condDead == condCyclic == condSaturated == False:
        label = "Irregular attractor"
        
    return label

# New version of test_activity
def classify_activity_attractors(A_trial, N, nb_chunks=4):
    length_A = len(A_trial)
    chunk_length = int(length_A/nb_chunks)
    label_full_length = test_attractor(A_trial, N)
    label_chunks = []
    
    # Check for non-trivial states
    for i in range(nb_chunks):
        A_chunk = A_trial[i*chunk_length:(i+1)*chunk_length]
        label_chunks.append(test_attractor(A_chunk, N))
    label = np.unique(label_chunks)
    if len(label) != 1:
        label = "Non trivial activity"
    else:
        label = label_full_length
    return label

# Legacy
def attractor_classification_old(A, N, intermediateTimeStep=1000, maxlag=800, test=False):
    condConst = False
    condDead = False
    condNonStat = False
    condPeriod = False
    condSaturated = False
    # Test for dead case
    A_trial = np.array(A)
    A_trial = A_trial[intermediateTimeStep+1:]
    diffA = np.diff(A_trial, n=1)
    if np.mean(A_trial) == 0:
        label = "No activity"
        condDead = True
    elif np.mean(A_trial) == 1:
        label = "Saturated attractor"
        condSaturated = True
    # Test for constant activity
    elif diffA[diffA <= 1 / N].size == len(diffA) and not condDead:  # Compute first drivative
            label = "Fixed point attractor"
            condConst = True
    # Test for non stationarity
    elif not condConst and not condDead and not condSaturated:
        A_half = A_trial[int(len(A_trial)/4):]
        diffA_half = np.diff(A_half, n=1)
        if diffA_half[diffA_half <= 1 / N].size == len(diffA_half) or np.mean(A_half) == 0:
            condNonStat = True
            label = "Non trivial activity"
    if not (condConst or condDead or condNonStat):
        # Test for periodicity
        # Compute Autocorrelation of the signal
        autoCorr, lagVector = fastCrossCorr(A_trial, norm=True, maxlag=maxlag)
        idxZero = np.where(lagVector == 0);
        idxZero = idxZero[0];
        idxZero = idxZero[0];
        autoCorr = np.array(autoCorr) 
        # ToDo : check why division by 0 happens some times
        autoCorrNorm = autoCorr / autoCorr[idxZero]  # Normalize with correlation at time lag 0
        autoCorrNorm = autoCorrNorm[idxZero:]
        idxPeriodMin = int(maxlag / 5)
        cutAutoCorr = autoCorrNorm[idxPeriodMin:]

        if test:
            plt.figure()
            plt.plot(autoCorrNorm)
            
        if len(cutAutoCorr[cutAutoCorr >= 0.9]) > 0:
            label = "Cyclic attractor"
            condPeriod = True
        elif not condPeriod:
            # 2nd test of periodicity
            # Autocorrelation of the autocorrelation
            corrOfCorr, lagVector = fastCrossCorr(autoCorr, norm=True, maxlag=maxlag)
            idxZero = np.where(lagVector == 0);
            idxZero = idxZero[0];
            idxZero = idxZero[0];
            corrOfCorr = corrOfCorr / corrOfCorr[idxZero]
            corrOfCorr = corrOfCorr[idxZero:]
            if test:
                plt.figure()
                plt.plot(corrOfCorr)
            cutAutoCorr = corrOfCorr[idxPeriodMin:]
            if len(cutAutoCorr[cutAutoCorr >= 0.95]) > 0:
                label = "Cyclic attractor"
                condPeriod = True

        if not condPeriod:
            # Test for non stationarity
            sizeTrial = len(A_trial)
            nbCut = 5
            moduloSize = sizeTrial % nbCut
            if moduloSize != 0:
                nbCut = moduloSize
            sizeCut = int(sizeTrial / nbCut)
            cutAverage = []
            cutEntropy = []
            for k in range(nbCut):
                # Compute average
                cutTrial = A_trial[k * sizeCut:(k + 1) * sizeCut - 1]
                cutAverage.append(np.mean(cutTrial))
                # Compute entropy
                values, counts = np.unique(cutTrial, return_counts=True)
                prob = counts / sum(counts)
                cutEntropy.append(entropy(prob, base=2))
            periodTested = False
            for k in range(1, nbCut):
                #for l in range(nbCut):
                    # if l == k:
                    #     continue
                    # # print('cond')
                    # ratioAvg = cutAverage[k] / cutAverage[l]
                    # ratioEntropy = cutEntropy[k] / cutEntropy[l]
 
                ratioAvg = cutAverage[k-1] / cutAverage[k]
                ratioEntropy = cutEntropy[k-1] / cutEntropy[k]
                if test:
                        print(cutEntropy[k-1], cutEntropy[k], ratioEntropy)
                        # print(cutAverage[k-1], cutAverage[k], ratioAvg)
                if ratioAvg < 0.65 or ratioAvg > 1.45 or ratioEntropy < 0.8 or ratioEntropy > 1.2:
                    condNonStat = True
                    label = "Non trivial activity"
                    break
                elif (ratioAvg < 0.8 or ratioAvg > 1.2 or ratioEntropy < 0.93 or ratioEntropy > 1.07) and not periodTested:
                    # Last test of periodicity
                    halfTime = int(len(A_trial) / 2) - 100
                    halfShrinkTrial = A_trial[halfTime:]
                    maxLag = len(halfShrinkTrial) - 100
                    autoCorr, lagVector = fastCrossCorr(halfShrinkTrial, norm=True, maxlag=maxLag)
                    idxZero = np.where(lagVector == 0);
                    idxZero = idxZero[0];
                    idxZero = idxZero[0];
                    autoCorr = autoCorr / autoCorr[idxZero]
                    autoCorr = autoCorr[idxZero:]
                    idxPeriodMin = int(maxLag / 5)
                    cutAutoCorr = autoCorr[idxPeriodMin:]
                    periodTested = True
                    if test:
                        plt.figure()
                        plt.plot(autoCorr)
                    if len(cutAutoCorr[cutAutoCorr >= 0.95]) > 0:
                        condNonStat = True
                        label = "Non trivial activity"
                        break
                if condNonStat:
                    break
            if test:
                print(cutEntropy)
                print(cutAverage)
                print('----')
            if not condNonStat and not condPeriod:
                label = "Irregular attractor"
            
    return label

#############################################################
# %% Excitatory-Inhibitory Balance
#############################################################

zeta = lambda x: sp.erf(1/(np.sqrt(2)*x))
    
def balance(mu, sigma):
    sigma_star = sigma/mu
    b = zeta(sigma_star)
    return b

def add_balance(df, name=None, path=None, N=None, K=None, 
                W=None, T=None, d=None, col='stdWeights'):

    # if 'balance' not in df.columns:
    if col not in df.columns:
        col = 'stdWeight'
    b = [balance(W, sigma, 'analytic') for sigma in df[col].values]
    df['balance'] = b
    if name is not None:
        print('SAVE', path, name)
        save_df(df, name, path, N=N, K=K, W=W, T=T, d=d)
        
    return df

#############################################################
# %% Simulations
#############################################################

def rescale(data):
    if len(data[data < 0]) > 0:
        data = data + abs(min(data))  # Remove negative values
    return data / max(data)  # Normalize data

def selectAllCircuits(nbcircuitPerSigma, N, K, nbtrial, stdWeightCond=None, I=0.2, meanWeight=-0.1, sim='statisticalAnalysis', experiment='globalAttractor'):
    # metadata folder
    meta_dir = dataPath(sim, experiment, N, K, meanWeight, I=I)
    metadataFiles = glob.glob(meta_dir +  '*metadata*_N' + str(N) + '*_T' + str(nbtrial) + '*_W' + str(meanWeight) + '*.npy')
    print(meta_dir +  '*metadata*_N' + str(N) + '*_T' + str(nbtrial) + '*_W' + str(meanWeight) + '*.npy')
    
    dictionaryArchitecture = {'metadata':[]}
    countPerSigma = 1
    
    # Randomly shuffle
    rnd.shuffle(metadataFiles)
    
    for i, metadataFile in enumerate(metadataFiles):
        
        # Load metadata
        try:
            metadata = np.load(metadataFile, allow_pickle='TRUE').item()
        except:
            print(f"ERROR: couldn't open the metadata:{metadataFile}")
            continue
        try:
            stdWeight = metadata['stdWeight']
        except:
            stdWeight = metadata['stdWeights']

        # Check that sigma is comprised in condition
        if stdWeightCond is None:
            pass
        elif len(stdWeightCond) == 2:
            if stdWeight < stdWeightCond[0] or stdWeight > stdWeightCond[1]:
                continue
        else:
            if stdWeight not in stdWeightCond:
                print(f'stdWeight {stdWeight} not in stdWeightCond!')
                continue

        # Update dictionary for the same value of sigma
        seed = metadata['seed']['seedWeights']
        if dictionaryArchitecture.get(stdWeight, None) is None:
            dictionaryArchitecture.update({stdWeight:[seed]})
            dictionaryArchitecture['metadata'].append(metadataFile)
            # print(seed, stdWeight)
        else:
            if len(dictionaryArchitecture[stdWeight]) == nbcircuitPerSigma:
                continue
            else:
                print(seed, stdWeight)
                dictionaryArchitecture[stdWeight].append(seed)
            dictionaryArchitecture['metadata'].append(metadataFile)
    return dictionaryArchitecture

#dictionaryArchitecture = selectAllCircuits(1, 10000, 16, 100)
#print(dictionaryArchitecture)

def selectCircuits(nbcircuit, nbtrial, label='after', sim='performanceTask', experiment='prunedCircuits', 
                   order='SA', size=(2000, 3000), bound=None):
    
    directory = dataPathPerf(sim, experiment)
    if label == '':
        label = 'labelNotpruned'
    elif label == 'after':
        label = 'labelPruned'
    filepath = directory+f'/dataDict_{label}_T{nbtrial}.npy' # _EA
    print(filepath)
    dataDic = np.load(filepath, allow_pickle='TRUE').item()

    if order == 'SA':
        order='meanTau'
    elif order == 'EA':
        order='tauAvgEA'
    
    circuitsToSelect = {}
    idx = 0
    tau_array = []
    orig_keys = []
    for i, (key, value) in enumerate(dataDic.items()):
        N = value['N']
        if N > size[0] and N < size[1]: # Reduce the range of network sizes
            orig_keys.append(key)
            circuitsToSelect.update({idx:dataDic.get(key)})            
            tau_array.append(circuitsToSelect[idx][order])
            idx += 1
    orig_keys = np.array(orig_keys)
    tau_array = np.array(tau_array)
    if nbcircuit < 100:
        if bound is not None:
            tau_array = np.array(tau_array)
            minLim = min(tau_array[tau_array>=bound[0]])
            maxLim = max(tau_array[tau_array<=bound[1]])
        else:
            minLim = min(tau_array)
            maxLim = max(tau_array)
        tau_selecteds = np.linspace(minLim, maxLim, nbcircuit)
        keys = np.unique([(np.abs(tau_array-tau_selected)).argmin() for tau_selected in tau_selecteds])
        print(tau_array[np.array(keys, dtype=int)])
        selectedCircuits = list( map(circuitsToSelect.get, keys) ) # Get list of dictionnaries of each selected keys
    if nbcircuit >= 100 :
        if bound is not None:
            tau_array = tau_array[tau_array > bound[0]]
            tau_array = tau_array[tau_array < bound[1]]
        keys = np.linspace(0, len(tau_array)-1, nbcircuit).astype(int)
        print(tau_array[np.array(keys, dtype=int)])
        selectedCircuits = list( map(circuitsToSelect.get, keys) ) # Get list of dictionnaries of each selected keys
    keys = np.array(keys, dtype=int)
    return selectedCircuits, orig_keys[keys]



#############################################################
# %% Saving DF
#############################################################

def find_duplicate(numbers):
    duplicates = [number for number in numbers if numbers.count(number) > 1]
    return list(set(duplicates))

def save_df(filePath, dic_result, index_name=None, save=True):
    try:
        df = pd.read_csv(filePath + '.csv')
        keys = [key for key in dic_result.keys()]
        if len(dic_result['stdWeights']) < len(df['stdWeights']):
            for key in dic_result.keys():
                if key == 'stdWeights':
                    continue
                df[key] = np.nan
                print('--------->', len(dic_result['stdWeights']), len(dic_result[key]))
                if len(df.loc[df['stdWeights'].isin(dic_result['stdWeights']), :]) > len(dic_result[key]):
                    print('Boooo')
                    print(len(df.loc[df['stdWeights'].isin(dic_result['stdWeights']), :]), len(dic_result[key]))
                    duplicates = find_duplicate(df['stdWeights'].values.tolist())
                    print('duplicates', duplicates)
                    values_to_del = list(set(duplicates).intersection(list(dic_result['stdWeights'])))
                    print('values_to_del', values_to_del)
                    idx = df.index[df['stdWeights'].isin(values_to_del)].tolist()
                    print('idx', idx, values_to_del)
                    df=df.drop(idx[0])
                    print('Yoooo', len(df), df['stdWeights'], dic_result['stdWeights'])
                if len(df.loc[df['stdWeights'].isin(dic_result['stdWeights']), :]) < len(dic_result[key]):
                    print(len(df), len(df.loc[df['stdWeights'].isin(dic_result['stdWeights']), :]), len(dic_result[key]))
                    df1 = pd.DataFrame(dic_result)
                    sigmas = list(df['stdWeights'].values)
                    sigmas.extend(df1['stdWeights'].values)
                    df2 = pd.concat([df, df1], axis=1, join='outer')
                    sigmas = np.unique(np.array(sigmas))
                    print(len(df2), len(sigmas))
                    df = df2.loc[:, ~df2.columns.duplicated(keep='first')]
                    df['stdWeights'] = sigmas
                else:
                    df.loc[df['stdWeights'].isin(dic_result['stdWeights']), key] = dic_result[key]
        elif len(dic_result['stdWeights']) > len(df['stdWeights']): 
            df1 = pd.DataFrame(dic_result)
            df = pd.concat([df, df1], axis=1, join='outer')
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
        else:
            df_new = pd.DataFrame.from_dict(dic_result)
            # df_new = df_new.reset_index()
            # df_new = df_new.set_index(['stdWeights'])
            for key in keys:
                if key not in df.columns:
                    df = df.join(df_new[key], lsuffix='_caller', rsuffix='_other')
                elif key != 'stdWeights' and key != 'index' :
                    df[key] = df_new[key].values
                    # print('before', df.columns)
    except:
        print("Oops!", sys.exc_info()[0], "occured.")
        print(">>", sys.exc_info()[1])
        df = pd.DataFrame.from_dict(dic_result)
    
    
    for col in df.columns:
        if 'Unnamed' in str(col):
            df.pop(col)
        elif '.' in str(col):
            df.pop(col)
            
    if index_name is not None:
        df.index.names = [index_name]

    if save:
        # Remove if duplicate columns
        df.to_csv(filePath + '.csv')
        
    return df

def name_format(N, K=None, W=None, d=None, T=None, sigma=None):
    spec_activities = ""
    if N is not None:
        spec_activities += f"_N{N}"
    if K is not None:
        spec_activities += f"_K{K}"
    if W is not None:
        spec_activities += f"_W{W}"
    if sigma is not None:
        spec_activities = f"_std{sigma}"
    if d is not None:
        spec_activities += f"_d{d}"
    if T is not None:
        spec_activities += f"_T{T}"
    
    return spec_activities

def load_df(name, data_path, N=None, K=None, W=None, T=None, d=None, sigma=None):
    spec_activities = name_format(N=N, K=K, W=W, d=d, T=T, sigma=sigma)
    full_name = name + spec_activities
    full_path = os.path.join(data_path, full_name)
    print(full_path)
    df = pd.read_csv(full_path + '.csv')
    return df

# %% main

def main():
    pass
    
if __name__ == '__main__':
    main()

