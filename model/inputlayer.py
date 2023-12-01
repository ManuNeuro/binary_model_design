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
'''
Input layer of the binaryModel
'''

#from snoop import snoop
import numpy as np
from typing import Union
from binary_model.utils import instantiateArchitecture, instantiateWeights, instantiateInputs
from binary_model.utils import generateRandomArchitecture, generateRandomWeights


class inputLayer():
    def __init__(self, N: int, inSize: int, weights=None, architecture=None):
        self.N = N
        self.inSize = inSize
        self.Ain = None
        self.Win = None
        self.inputLayer = True
        self.u = np.zeros((1, 1))
        self.Win_matrix =  np.zeros((1, 1))
        # self.metadata = {}
        
        if architecture is not None:
            self.setArchitecture(architecture)
        
        if weights is not None:
            self.setWeightsIn(weights)
    
    def setArchitectureIn(self, 
                          architecture: Union[None, list, np.ndarray], 
                          **kwargs):
        # If no architecture is provided
        if architecture is None:
            architecture = generateRandomArchitecture(self, layer='input', **kwargs)
        
        # Initialize architecture
        self.Ain, _ = instantiateArchitecture(architecture, Nin=self.inSize, Nout=self.N)
            
    def setWeightsIn(self, 
                     weights: Union[None, list, np.ndarray], 
                     option='sparse', 
                     **kwargs):
        # If no weights is provided
        if weights is None:
            weights = generateRandomWeights(self, layer='input', **kwargs)    
        
        # Arguments for weight initialization
        if self.Ain is not None:
            print('SetWeightOut', 'A is not None')
            kwargs.update({'A':self.Ain})
            kwargs.update({'Nin':self.inSize,
                   'Nout':self.N,
                   'option':option})
        else:
            raise Exception('Error! Ain must be initialized to set weights')
        
        # Initialize weights
        self.Win_matrix , self.Win, _ = instantiateWeights(weights, **kwargs)
        if kwargs.get('distribution', None) is not None:
            self.metadata.update({'input_distribution':kwargs['distribution']})
        if kwargs.get('sigma', None) is not None:
            self.metadata.update({'input_mu':kwargs['mu'],
                                  'input_sigma':kwargs['sigma']})
              
    def setInputIn(self,
                    u: Union[list, np.ndarray],
                    timings: Union[list, np.ndarray], 
                    duration: int,
                    type_='sparse',
                    tag=None,
                    **kwargs):            
        self.inputCond = True
        
        # If no architecture and weights are initialized 
        if self.Ain is None:
            print('build input')
            I = kwargs.get('I', 0.6)
            seed = kwargs.get('seed', None)
            distribution = kwargs.get('distribution', 'gaussian')
            option = kwargs.get('option', 'global')
            self.initRandomArchitectureIn(option, I, seed)
            self.initRandomWeightsIn(distribution, seed, **kwargs)
        
        # Initialize input stream
        u, t = instantiateInputs(u, 
                                timings=timings, 
                                Nin = self.inSize,  
                                Nout = duration, 
                                **{'option':type_})
        self.inputStream= {'u':u,
                           't':t,
                           'type':type_}

        # Update metadata
        if tag is not None:
            self.metadata.update({'input':tag})