##Function remains unchanged ##
import torch.nn as nn
import math
import torch

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.hr2hr = nn.Linear(hidden_size,hidden_size,bias=bias)
        self.data = None
        self.Horiz = True #whether it is horizontal evluaiton or verical
        self.Prev_T = None
        #print("Input size:",self.input_size)
        #print("Hidden size:",self.hidden_size)

        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x,hidden):

        #print("x shape inside gru:",x.shape)
        #print("hidden shape inside gru:",hidden.shape)
        #print("horr",hidden.shape)
        x = x.view(-1, x.size(1))
        #print("horr",x.shape)
        gate_x = self.x2h(x)
        gate_x = gate_x.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)

        #i_r, i_i, i_n = self.data[0],self.data[1],self.data[2]
        gate_h = self.h2h(hidden)
        gate_h = gate_h.squeeze()
        h_r, h_i = gate_h.chunk(2, 1)
        #print("horr gate_h0=",gate_h.shape)

        resetgate = torch.sigmoid(i_r + h_r)#r
        inputgate = torch.sigmoid(i_i + h_i)#z


        hr = self.hr2hr(resetgate*hidden)
        #print("resetgate,hidden",resetgate.shape,hidden.shape,hr.shape)
        newgate = torch.tanh(i_n + hr)#c

        #hy = newgate + inputgate * (hidden - newgate)
        #hy = (1-z)*c + z*h
        #hy = (1-inputgate)*newgate + inputgate*hidden

        #ResNet
        #return (1-inputgate)*(newgate - hidden)
        return (1-inputgate)*newgate + inputgate*hidden