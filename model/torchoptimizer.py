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
# Date: July 12th, 2021
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada
'''
Pytorch optimizer for training the readout of binaryModel

NB: this is the optimizer used for the training in the final version of the paper.
'''
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

class torchoptimizer(nn.Module):
    def __init__(self, inSize, outSize, activation=None, loss=None, optim=None):
        super(torchoptimizer, self).__init__()
        self.linear = nn.Linear(inSize, outSize)
        self.inSize = inSize
        self.outSize = outSize
        self.optim = None
        if loss is not None:
            self.setLoss(loss)
        if optim is not None:
            self.setOptimizer(optim)
        if activation is not None:
            self.setActivation(activation)
            
    def setActivation(self, activation):
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        if activation == 'identity':
            self.activation = torch.nn.Identity
    
    def setLoss(self, loss):
        if loss == 'MSE':
            self.loss = nn.MSELoss()
        elif loss == 'cross-entropy':
            self.loss = nn.CrossEntropyLoss()
        elif loss == 'hinge':
            self.loss = nn.HingeEmbeddingLoss()
        elif loss == 'BCE':
            self.loss = nn.BCELoss()
    
    def setOptimizer(self, optim):
        if optim == 'SGD':
            self.optim = torch.optim.SGD
        elif optim == 'adam':
            self.optim = torch.optim.Adam
        elif optim == 'sparseAdam':
            self.optim = torch.optim.SparseAdam
        elif optim == 'RMS':
            self.optim = torch.optim.RMSprop
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
    
    def predict(self, X_test, y_test, task='classification'):
        with torch.no_grad():
            y_predicted = self.forward(X_test)
            if task == 'classification':
                y_predicted = np.round(y_predicted)
                acc = y_predicted.eq(y_test).sum() / float(y_test.shape[0])
                print(f'accuracy = {acc:.4f}')
            elif task == 'regression':
                err = self.loss(y_predicted, y_test) 
                print(f'Error = {err:.6f}')
        return y_predicted
    
    def train(self, X, Y, epoch, alpha=1E-4, **kwargs):
        
        scheduler = kwargs.get('scheduler', False)
        kwargs.pop('scheduler', None)
        
        # Initialize the optimizer
        self.optim = self.optim(self.parameters(), lr=alpha, **kwargs)
        if scheduler:
            self.scheduler = []
            print('-- Scheduler loaded ...')
            self.scheduler.append(torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=0.01, 
                                                                      total_steps=None, epochs=epoch, 
                                                                      steps_per_epoch=2, pct_start=0.3, 
                                                                      anneal_strategy='cos', cycle_momentum=True, 
                                                                      base_momentum=0.85, max_momentum=0.95, div_factor=25.0, 
                                                                      final_div_factor=10000.0, three_phase=False, last_epoch=-1, verbose=False))
            # self.scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, epoch, 
            #                                                             eta_min=0, last_epoch=-1, 
            #                                                             verbose=False))
            # self.scheduler.append(torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 
            #                                                                  mode='min', 
            #                                                                  factor=0.1, 
            #                                                                  patience=10, 
            #                                                                  threshold=0.0001, 
            #                                                                  threshold_mode='rel', 
            #                                                                  cooldown=0, 
            #                                                                  min_lr=0, eps=1e-08, verbose=False))
        # Training
        loss_checker = 0
        for k in range(epoch):
            
            # Forward pass (prediction)
            y_pred = self.forward(X)
            
            # Compute loss
            l = self.loss(y_pred, Y)
            
            # Backward pass (gradient)
            l.backward()
            
            # Update weights
            self.optim.step()
            
            # Reset gradients
            self.optim.zero_grad()
            
            # Scheduler 
            if scheduler:
                #valid_loss = l.item() * X.size(0)
                self.scheduler[0].step()
                #self.scheduler[1].step(valid_loss)

            if (k+1) % 100 == 0:
                if loss_checker == l.item():
                    break
                loss_checker = l.item() 
        print(f'Training converged after {k} number of epoch')
        layer = self.linear.state_dict()
        self.w = layer['weight'].numpy()
        self.b = layer['bias'].numpy()
    
    
def testReadout(X, Y, nbtrials, **optionTrain):
    test_size = optionTrain.get('testSize', 0.3)
    epoch = optionTrain.get('epoch', 10000)
    activation = optionTrain.get('activation', 'sigmoid')
    alpha = optionTrain.get('alpha', 0.001)
    loss = optionTrain.get('loss', 'MSE')
    optim = optionTrain.get('optimizer', 'adam')
    task = optionTrain.get('task', 'regression')
    scheduler = optionTrain.get('scheduler', False)
    nbtrials = optionTrain.get('nbTrial', 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=False)
    
    # Initialize tensor
    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    
    X_train  = X_train.view(X_train.shape[0], 1)
    X_test  = X_test.view(X_test.shape[0], 1)
    y_train  = y_train.view(y_train.shape[0], 1)
    y_test  = y_test.view(y_test.shape[0], 1)
        
    # Model
    model = torchoptimizer(1, 1)
    model.setActivation(activation)
    model.setLoss(loss)
    model.setOptimizer(optim)
    
    err_trials = []
    for k in range(nbtrials):
        model.train(X_train, y_train, alpha=alpha, epoch=epoch, **{'scheduler':scheduler})
        y_pred = model.predict(X_test, y_test, task=task)
        err = (y_pred - y_test)**2
        err_trials.append(err)
    return  err_trials
    
    