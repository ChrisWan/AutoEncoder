# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:26:48 2020

@author: wanke
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding ='latin-1')
users =  pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding ='latin-1')
rating =  pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding ='latin-1')

# Preparing the training and test sets
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting number of all user and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Creating the matrix through an array <- possible but for torch list of lists
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)

# Convert data into tensors
training_set = torch.FloatTensor(training_set) # .cuda() added to run it on GPU
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20) # 20 is output and experimental -> tuning, 20 features of movies e.g. genre
        self.fc2 = nn.Linear(20, 10) # 20 neurons -> 10 neurons
        self.fc3 = nn.Linear(10, 20) # start to decode 10 -> 20 -> nb_movies
        self.fc4 = nn.Linear(20, nb_movies) # get same structured vector as in input vector
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
    
sae = SAE()
# sae.cuda() # run on GPU
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the network
nb_epochs = 200
for epoch in range(1, nb_epochs + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if (torch.sum(target.data > 0) > 0): # evtl. .cuda()
            output = sae(input)
            target.requires_grad = False
            output[target == 0] == 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # evtl. .cuda()
            loss.backward()
            train_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
            optimizer.step() # backward = direction in which the weights are updated, step is by how much
    print('epoch: ' + str(epoch) + ' loss: ' + str(float(train_loss/s)))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = Variable(test_set[id_user]).unsqueeze(0)
        if (torch.sum(target.data > 0) > 0): # evtl. .cuda()
            output = sae(input)
            target.requires_grad = False
            output[target == 0] == 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # evtl. .cuda()
            test_loss += np.sqrt(loss.item()*mean_corrector)
            s += 1.
print('loss: ' + str(test_loss/s))


