import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Wave(nn.Module):
    def __init__(self,gru_1,gru_2,hidden_size,times = 1,batch_size = 256,gru_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.gru_1 = gru_1 ##nn.GRU(self.hidden_size,self.hidden_size,self.gru_layers)
        self.gru_2 = gru_2
        self.ts = times
        self.batch_size = batch_size

    def forward(self,hidden_state):
        
        x_t = hidden_state[self.batch_size:hidden_state.shape[0]//2,:]
        h_t = hidden_state[hidden_state.shape[0]//2:-self.batch_size,:]
        h_t_1 = hidden_state[hidden_state.shape[0]//2:,:]
        x_t_1 = hidden_state[:hidden_state.shape[0]//2,:]
        
        #print("Shape of x_t:",x_t.shape)
        #print("Shape of h_t:",h_t.shape)
        
        x_t_2 = hidden_state[:hidden_state.shape[0]//2 - self.batch_size,:]
        h_t_2 = hidden_state[hidden_state.shape[0]//2 + self.batch_size:,:]

    
        t = self.gru_1(x_t,h_t)[:-self.batch_size] + (1/self.ts)*(h_t_1[:-2*self.batch_size,:] -2*h_t_1[1*self.batch_size:-1*self.batch_size,:] 
                                             + h_t_1[2*self.batch_size:,:]) + (2*h_t_1[1*self.batch_size:-1*self.batch_size,:] 
                                             - x_t_1[1*self.batch_size:-1*self.batch_size,:]) + self.gru_2(x_t_2,h_t_2)[self.batch_size:]
        
        #print("Shape of t:",t.shape)
        #print("Shape of t1:",t_1.shape)
        
        #print(t[:self.batch_size].shape)
        #print(t[-self.batch_size:].shape)
        
        dfdt = torch.concat([t[:self.batch_size],t,t[-self.batch_size:]],axis = 0)
        dfdt = torch.concat([h_t_1,dfdt],axis = 0)
        #print(dfdt.shape)
        return dfdt
        
        
        
        
        
