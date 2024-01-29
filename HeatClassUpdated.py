import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Heat(nn.Module):
    def __init__(self,gru,hidden_size,times = 1,batch_size = 256,gru_layers = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_layers = gru_layers
        self.gru = gru ##nn.GRU(self.hidden_size,self.hidden_size,self.gru_layers)
        self.ts = times
        self.batch_size = batch_size

    def forward(self,hidden_state):
    
        x_t = hidden_state[self.batch_size:hidden_state.shape[0]//2,:]
        h_t = hidden_state[hidden_state.shape[0]//2:-self.batch_size,:]
        h_t_1 = hidden_state[hidden_state.shape[0]//2:,:]
        #x_t_1 = hidden_state[:hidden_state.shape[0]//2,:]
        #print("self ts shape:",self.ts.shape)
        #print("Shape of x_t:",x_t.shape)
        #print("Shape of h_t:",h_t.shape)
        #print("Shape of ts:",self.ts.shape)
        t = self.gru(x_t,h_t)[:-self.batch_size] + (1/self.ts)*(h_t_1[:-2*self.batch_size,:] -2*h_t_1[1*self.batch_size:-1*self.batch_size,:] 
                                             + h_t_1[2*self.batch_size:,:])
        #(2*h_t_1[1*self.batch_size:-1*self.batch_size,:]  - x_t_1[1*self.batch_size:-1*self.batch_size,:]) 
        
        #print("Shape of t:",t.shape)
        
        #print(t[:self.batch_size].shape)
        #print(t[-self.batch_size:].shape)
        
        dfdt = torch.concat([t[:self.batch_size],t,t[-self.batch_size:]],axis = 0)
        dfdt = torch.concat([h_t_1,dfdt],axis = 0)
        #print(dfdt.shape)
        return dfdt
        
        
        
        
        
