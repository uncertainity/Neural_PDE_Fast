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
from WavePlusPlus import Wave
from NeuralPDEPlusPlus import NeuralPDE
import torchdyn
import logging

print("Torch version:",torch.__version__)
print("Torchdyn version:",torchdyn.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:",device)
solver = "dopri5"
size = 64
default = 200
lr = 0.005
steps = 2
dataset_name = "walk"
epochs = 100
model = "Wave"

logging.basicConfig(level = logging.INFO,format = '%(asctime)s %(message)s',filename = "wave.log",filemode="w")
logging.info("Logging has started")
## declare the input,hidden,output size and the pre and post neural networks ##

def load_dataset(dataset_name):
    
    #if dataset_name == "walk":
    print("I am in load_dataset class")
    dataset = Walker2dImitationData(seq_len = 64)
    train_x = torch.Tensor(dataset.train_x)
    train_y = torch.LongTensor(dataset.train_y)
    train_ts = torch.Tensor(dataset.train_times)
    
    test_x = torch.Tensor(dataset.test_x)
    test_y = torch.LongTensor(dataset.test_y)
    test_ts = torch.Tensor(dataset.test_times)
    
    valid_x = torch.Tensor(dataset.valid_x)
    valid_y = torch.LongTensor(dataset.valid_y)
    valid_ts = torch.Tensor(dataset.valid_times)
    
    print(train_y.shape,test_y.shape)
    
    train = data.TensorDataset(train_x, train_ts, train_y)
    test = data.TensorDataset(test_x, test_ts, test_y)
    valid = data.TensorDataset(valid_x, valid_ts, valid_y)
    return_sequences = True
	    
    trainloader = data.DataLoader(train, batch_size=256, shuffle=True)
    testloader = data.DataLoader(test, batch_size=256, shuffle=False)
    in_features = train_x.size(-1)
    validloader = data.DataLoader(valid, batch_size=256, shuffle=False)
    num_classes = 17
    
    return trainloader, testloader, validloader , in_features, num_classes, return_sequences,train_x.shape[1]
        
        

# def load_dataset(dataset_name):
# 	if dataset_name == "person":
# 		dataset = PersonData()
# 		train_x = torch.Tensor(dataset.train_x)
# 		train_y = torch.LongTensor(dataset.train_y)#[:,0]
# 		train_ts = torch.Tensor(dataset.train_t)
# 		test_x = torch.Tensor(dataset.test_x)
# 		test_y = torch.LongTensor(dataset.test_y)#[:,0]
# 		test_ts = torch.Tensor(dataset.test_t)
# 		train = data.TensorDataset(train_x, train_ts, train_y)
# 		test = data.TensorDataset(test_x, test_ts, test_y)

		    	
# 		print(train_y.shape,test_y.shape)
# 		return_sequences = True
# 	elif dataset_name == 'walk':
# 		dataset =  Walker2dImitationData(seq_len=64)
# 		train_x = torch.Tensor(dataset.train_x)
# 		train_y = torch.LongTensor(dataset.train_y)#[:,0]
# 		train_ts = torch.Tensor(dataset.train_times)
# 		test_x = torch.Tensor(dataset.test_x)
# 		test_y = torch.LongTensor(dataset.test_y)#[:,0]
# 		test_ts = torch.Tensor(dataset.test_times)
# 		valid_x = torch.Tensor(dataset.valid_x)
# 		valid_y = torch.LongTensor(dataset.valid_y)#[:,0]
# 		valid_ts = torch.Tensor(dataset.valid_times)


# 		train = data.TensorDataset(train_x, train_ts, train_y)
# 		test = data.TensorDataset(test_x, test_ts, test_y)
# 		valid = data.TensorDataset(valid_x, valid_ts, valid_y)
# 		return_sequences = True

# 		print(train_y.shape,test_y.shape,valid_y.shape)
# 	trainloader = data.DataLoader(train, batch_size=256, shuffle=True)
# 	testloader = data.DataLoader(test, batch_size=256, shuffle=False)
# 	in_features = train_x.size(-1)
# 	if dataset_name == 'walk':
#         print("Entering load_dataset")
# 		validloader = data.DataLoader(valid, batch_size=256, shuffle=False)   	
# 		num_classes = 17
# 		return trainloader, testloader, validloader , in_features, num_classes, return_sequences,train_x.shape[1]
# 	else:
# 		num_classes = int(torch.max(train_y).item() + 1)
# 		return trainloader, testloader, in_features, num_classes, return_sequences,train_x.shape[1]




input_size = 17  ##----> input dimension of the vector
hidden_size = 64  ##---> dimension of the pde/ode
output_size = 17  ##----> dimension of output

# pre_nn = PreNeuralNetwork(input_size,hidden_size)
# post_nn = PostNeuralNetwork(hidden_size,output_size)
# ## Heat equation to be declared ##
# heat = Heat(hidden_size)

neuralpde = NeuralPDE(input_size,hidden_size,output_size,model)
neuralpde.to(device)

criterion = nn.MSELoss()
params = list(neuralpde.parameters())
optimizer = optim.RMSprop(params,lr = lr)


def feed_forward(loader,num_classes,train = True,return_sequences = True,seq_len = 64):
    
    if return_sequences:
        Y_hat = torch.zeros((0,seq_len,num_classes))
    else:
        Y_hat = torch.zeros((0,num_classes))

    Y = torch.zeros((0)).type(torch.int)

    if train:
        print("Entering with train mode")
        neuralpde.train()
    else:
        print("Entering eval mode")
        neuralpde.eval()

    Loss = 0
    k = 0
    ## Now put the dataset in the format of [batch_size,seq_len,seq_size] ##
    for batch in loader:
        train_x,train_ts,train_y = batch
        mask = None

        optimizer.zero_grad()

        batch_size = train_x.shape[0]
        seq_len = train_x.shape[1]
        seq_size = hidden_size
        #print("batch_size:",batch_size)
        #print("seq len:",seq_len)
        #print("seq size:",seq_size)
        ## The data is stored is batch_size,seq_len,seq_size format with for the current step --> initiallly step 0
        ## Seq_len + 2 is taken to easily calculate the numerical difference term as a matrix
        hidden_state = torch.zeros((batch_size*seq_len + steps*batch_size,seq_size),device = device)
        #print(batch_size*seq_len + steps*batch_size)
        #print("hidden state shape:",hidden_state.shape)

        #print("Hidden state device:",hidden_state.device)
        #init_hidden_state_zero = torch.zeros((batch_size,seq_len + 2,seq_size),device = device)
        Times = torch.zeros((batch_size*seq_len),device = device)
        #print("Batch size:",batch_size)
        for j in range(seq_len):
            input = train_x[:,j].to(device)
            input = neuralpde.pre_nn(input)
            ts = train_ts[:,j].to(device)
            #print(ts.shape)
            hidden_state[(j+1)*batch_size:(j+2)*batch_size] = input
            #print((j)*batch_size,(j+1)*batch_size)
            #print(((ts-ts.min())/(ts.max()-ts.min())).shape)
            Times[j*batch_size:(j+1)*batch_size] = ((ts-ts.min())/(ts.max()-ts.min()))[:,0]+1
            if j == 0:
                hidden_state[:batch_size] = input
            elif j == seq_len - 1:
                hidden_state[-batch_size:] = input
        
        #print("Hidden_state size:",hidden_state.shape)
        #asdad += 1            
        Times[Times.isnan()] = 1.0
        Times = Times.reshape([Times.shape[0],1])
        #print("Times shape inside forward:",Times.shape)
        #print("hidden state shape:",hidden_state.shape)
        ## for the estimate the first hidden state, the zero-th step is repeated twice. for the second state, we will take the zero-th and
        ## first step and so on..
        init_hidden_state = torch.cat([hidden_state,hidden_state],axis = 0).to(device)
        #neuralpde.h = init_hidden_state
        #neuralpde.batch_size = batch_size
        new_hidden_state = neuralpde(init_hidden_state,batch_size,Times.to(device))
        y_ = new_hidden_state[new_hidden_state.shape[0]//2 + batch_size:-batch_size]
        #print(y_.shape)
        
        outputs = []
        
        for j in range(seq_len):
            output = neuralpde.post_nn(y_[j*batch_size:(j+1)*batch_size])
            outputs.append(output)
        y_hat = torch.stack(outputs,dim = 1).to(device)
        #print(y_hat.shape)
        #print(train_y.shape)  
        #error += 1
        y_hat = torch.stack(outputs,dim = 1).to(device)
        train_y = train_y.type(y_hat.type())

        Y_hat = torch.cat((Y_hat,y_hat.detach().cpu()))
        Y = torch.cat((Y,train_y.detach().cpu()))

        loss = criterion(y_hat,train_y.to(device))

        if train:
          if k % 500 == 0:
            print("k = ", k)
            print(loss.item())
          k += 1
          loss.backward()
          # for params in neuralpde.parameters():
          #   print(params.grad.shape)
          #   print(params.grad)
          #   print("-"*20)
          # print("*"*40)
          optimizer.step()

        Loss += loss.item()

        #del neuralpde.h
        #torch.cuda.empty_cache()
    loss = criterion(Y_hat,Y)
    print("Y_hat shape:",Y_hat.shape)
    print("Y shape:",Y.shape)
    if train:
        return loss.item()
    else:
        return loss.item(),Y_hat,Y

train_loss = []
train_acc = []
test_loss = []
test_acc = []

Best_loss = 10000
Best_loss_train = 10000
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones = [100],gamma = 0.1)

print("Original place where train and test and valid loaders are being called")
trainloader, testloader, validloader, in_features, num_classes, return_sequences,seq_len = load_dataset("walk")



for epoch in range(epochs):
    print("\nEpoch:",epoch,seq_len)
    if dataset_name == "walk":
        loss = feed_forward(trainloader,num_classes,train = True)
        print("Train loss:",loss)
        train_loss.append(loss)
        loss,_,_ = feed_forward(validloader,num_classes,train = False)
        print("Valid loss:",loss)
        test_loss.append(loss)
        print("*"*20)
        logging.info(f"Epoch: {epoch} - Train Loss:{train_loss[-1]}; Valid Loss: {test_loss[-1]}")
        logging.info(f"Best Validtaion loss: {Best_loss}, Best train loss:{Best_loss_train}")    
       
    if loss < Best_loss:
        Best_loss = loss
        Best_loss_train = train_loss[-1]
        #torch.save(neuralpde.state_dict(), "/content/drive/MyDrive/wave_models/"+dataset+"heat_arka_best_steps%d_%s_model.pth"%(steps,solver))
        print("Best validation loss:",Best_loss," then train loss:",Best_loss_train, "\n")
    scheduler.step()
    #np.save("/content/drive/MyDrive/wave_models/"+dataset+"heat_steps%d_%s_train_loss.npy"%(steps,solver),train_loss)
    #np.save("/content/drive/MyDrive/wave_models/"+dataset+"heat_steps%d_%s_valid_loss.npy"%(steps,solver),test_loss)
if dataset_name =='walk':
	#neuralpde.load_state_dict(torch.load("./models/wave_plus_plus"+"%d_best_steps%d_%s_model.pth"))
	loss =feed_forward(testloader,train=False)
	print("Test loss:",loss)

