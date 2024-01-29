import os
import argparse
from irregular_sampled_datasets import PersonData,Walker2dImitationData
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional import accuracy
import time
import numpy as np
import torch
import math
from torchdyn.core import NeuralDE
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint
from itertools import chain
from GRUCell import GRU
from HeatClassUpdated import Heat
from WaveClassUpdated import Wave

class PreNeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.model = torch.nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
        )

    def forward(self,input):
        return self.model(input)


class PostNeuralNetwork(nn.Module):
    def __init__(self,hidden_size,output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = torch.nn.Sequential(
            nn.Linear(self.hidden_size,self.output_size),
        )
    def forward(self,input):
        return self.model(input)


class NeuralPDE(nn.Module):

    def __init__(self,input_size,hidden_size,output_size,model = "Wave"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = GRU(self.input_size,self.hidden_size)
        
        if model == "Wave":
            self.pdefunc = Wave(self.gru,self.hidden_size)
        elif model == "Heat":    
            self.pdefunc = Heat(self.gru,self.hidden_size)
        
        self.pre_nn = PreNeuralNetwork(self.input_size,self.hidden_size)
        self.post_nn = PostNeuralNetwork(self.hidden_size,self.output_size)
        self.ivpnet = None
        print("Initializing the neural DE")
        self.NODE = NeuralDE(self.pdefunc,solver = "dopri5",atol = 1e-3,rtol = 1e-3)
        self.batch_size = None
        self.h = None
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self,x0,batch_size,times,time_steps = torch.Tensor([0.,1.]),solver = "dopri5"):
        #print("I am inside the neural pde func")
        self.pdefunc.ts = times
        self.pdefunc.batch_size = batch_size
        return self.NODE.trajectory(x0,time_steps)[1]
    



