# this script is to read data from the trigger_events parquet file and pass them to a RNN model 

import numpy as np 
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt

import wandb
#import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import RNN
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader  
import torch.optim as optim
from torchsummary import summary

device = torch.device("cuda:0")

# start a new wandb run to check the script
wandb.init(
    # set the wandb project where this run will be logged
    project="777_training",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.1, # here i put the initial lr 
    "architecture": "RNN",
    "epochs": 8,
    "batch_size": 16
    }
)


# read data from the parquet file
dataset = ak.from_parquet("/n/home06/qhu/tau_sim/777_trigger_events.parquet") 
#dataset = ak.from_parquet("111_trigger_events.parquet") 

X_data = []
Y_data = []

for i in dataset:
    amp_list = []
    for y in i["amplitudes"]:
        amp_list.append((y))
    X_data.append(amp_list)  # (one)feature
    Y_data.append(i["label"]) # target value 


# find the max. no. of timesteps
timestep = []
for i in X_data:
    timestep.append(len(i))

pad_len = max(timestep)

# pad the list before turning into numpy array
X_data_padded = []

for i in X_data:
    i = i + [0.0] * (pad_len - len(i))
    X_data_padded.append(i)


# Convert to np array  
X_data_np = np.array(X_data_padded)
Y_data_np = np.array(Y_data) 


# notice that predictions that converges to 0.7 is a result of more significant no. of labels 1 than 0, as the initial loss is alr small
# hence a penalizing system to the loss should be introduced, i.e. the loss should be bigger if the network always predict 1
# 



# Convert data to tensors
X_data_tensor = torch.tensor(X_data_np, dtype = torch.float32) 
X_data_tensor_unsqueeze = torch.unsqueeze(X_data_tensor, -1) # 2D to 3D tensor
Y_data_tensor = torch.tensor(Y_data_np, dtype = torch.float32) 

# to check tensor size
print(f"X_data tensor has shape: {X_data_tensor.shape}")
print(f"X_data_tensor_unseqeeze has shape: {X_data_tensor_unsqueeze.shape}") 
print(f"Y_data_tensor has shape: {Y_data_tensor.shape}")

# normalize X data and Y data before passing to DataSet/DataLoader 
mean, std = torch.mean(X_data_tensor_unsqueeze), torch.std(X_data_tensor_unsqueeze)
X_data_tensor_normalized = (X_data_tensor_unsqueeze - mean) / std 


# check that X_data has been normalized 
print(f"the standard deviation and mean of X_data is:{torch.std_mean(X_data_tensor_unsqueeze)}")
print(f"the standard deviation and mean of normalized X_data is:{torch.std_mean(X_data_tensor_normalized)}")


# train-test split 
train_data, test_data, train_labels, test_labels = train_test_split(
    X_data_tensor_normalized, Y_data_tensor, test_size=0.2, random_state = 42) 


# load the data 
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle = False)

# show the shape of training_data after DataLoader
#for X_train, Y_train in train_loader[0]:
#    print(X_train.shape)
#    print(Y_train.shape)


# a prototype RNN model with 3 layers
BATCH_SIZE = 16 
N_STEPS = pad_len # where pad_len is trigger/quality event dependent 
N_INPUTS = 1
N_NEURONS = 16
N_OUTPUTS = 8
N_DIRECTION = 1
N_EPOCHS = 32


# a prototype RNN model with 3 layers (this script changes it to bidirectional to reduce feature loss)
class RNN_model(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs, n_directions):
        super(RNN_model, self).__init__()
        
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_directions = n_directions 
        
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons, 1, nonlinearity='relu') 
        self.FC = nn.Linear(self.n_neurons, self.n_outputs) 
        self.final_fc = nn.Linear(in_features=n_outputs, out_features=1, bias=True)


    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        return (torch.rand(1, self.batch_size, self.n_neurons))  # can try with different init_hidden to disturb initial weights

      
    def forward(self, X):
        X = X.permute(1, 0, 2) 
        #print(X.shape)
        self.batch_size = X.size(1)
        self.hidden = self.init_hidden() # to initialize hidden state for every loop
        #print(self.hidden.shape)
        
        # lstm_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each lstm_out)
        lstm_out, self.hidden = self.basic_rnn(X, self.hidden) 
        #print(lstm_out.shape)
        out = self.FC(self.hidden) 
        #print(out.shape)
        out = self.final_fc(out)  
        #print(out.shape)

        return out 



# initialize a model
model = RNN_model(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS, N_DIRECTION)
model = model.to(device) 
#loss_fn = nn.MSELoss()
weights = [0.65, 0.35] # i.e., more signals than background 
loss_fn = nn.BCELoss(weight = weights) # weights defining the frequency in the dataset in a mini batch 
optimizer = optim.SGD(model.parameters(), lr = 0.01) 
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode = 'min',
    patience = 3,
    factor = 0.7,
    eps = 1e-20,
    verbose = True
    )


# to have a sense of the size of the model 
for param in model.parameters():
    print(type(param), param.size())
    print(param.numel())

trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
)
print(f"the no. of trainable params of the model is: {trainable_params}") 


# summary of the model in terms of trainable parameters at each layer 
print(model) 
#summary(model, input_size = (pad_len, 1)) 


# model training 
for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    train_loss = 0
    train_acc = 0
    correct = 0 
    loss_list = []

    model.train()

    # TRAINING ROUND
    for i, data in enumerate(train_loader): 
        data = data.to(device)
         # zero the parameter gradients
        optimizer.zero_grad()
        
        # reset hidden states
        model.hidden = model.init_hidden()  
        model = model.hidden.to(device)  # return a copy only 

        inputs, labels = data # X_train, Y_train 
        #print(labels)
        #print(labels.shape) 
        #print(inputs.shape)

        outputs = model(inputs) 
        print(f"predicted outputs: {outputs}")
        print(f"actual labels: {labels}")
        outputs = outputs.view(-1) 
        #print(outputs.shape)
    
        #print(model.basic_rnn.bias_hh_l0)
        loss = loss_fn(outputs, labels)
        

        # sig/back weighted loss value
        #print(f"actual label is: {labels}")
        #if labels == 1:
        #    loss = 0.65 * loss_fn(outputs, labels) # the weight depending on the ratio of signal events in the training dataset 
        #else:
        #    loss = 0.35 * loss_fn(outputs, labels) 


        #print(loss)
        #print(model.basic_rnn.weight_hh_l0.grad)

        loss.backward() # compute the gradient of the loss with respect to model parameters =
        #print(model.basic_rnn.weight_hh_l0.grad)
        #print(model.basic_rnn.weight_hh_l0)
    

        optimizer.step() # call the step function to make an update on the model parameters 
        #print(model.basic_rnn.bias_hh_l0)

        #print(optimizer) 
        #print(loss)

        train_loss += loss  
        #if outputs == labels:
        #    correct += 1
        
        #transformed_outputs = nn.Sigmoid(outputs)
        transformed_outputs = (outputs>0.5).float()
        
        correct += (transformed_outputs == labels).float().sum()
        #print(f"no of correct prediction: {correct}")

    
        #metric = BinaryAccuracy(threshold=0.7)
        #metric.update(input, labels)
        #metric.compute()

    
    scheduler.step(loss)
    train_accuracy =  100 * correct / len(train_dataset)

    model.eval()

    # log metrics to wandb 
    wandb.log({
        "epoch" : epoch, 
        "train_accuracy": train_accuracy, 
        "train_loss": train_loss
        })

    # Log train and validation metrics to wandb
    #loss, train_accuracy
    #val_metrics = {"val/val_loss": loss,
    #                "val/val_accuracy": train_accuracy}
    #wandb.log({**metrics, **val_metrics})


    print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f' 
          %(epoch, train_loss / i, train_accuracy))

    #loss_list.append(train_loss.data)

    #print(model.basic_rnn.weight_hh_l0)
    #print(model.basic_rnn.bias_hh_l0)

    #print(model.FC.weight)
    #print(model.FC.bias)

    #print(model.final_fc.weight)
    #print(model.final_fc.bias)



    # model testing (also for each epoch as was for model training)
    test_correct = 0.0

    for i, data in enumerate(test_loader):
        data = data.to(device)
        inputs, labels = data 
        outputs = model(inputs)
        outputs = outputs.view(-1)
        print(outputs)
        print(labels)

        transformed_outputs = (outputs>0.5).float()
        test_correct += (transformed_outputs == labels).float().sum()



    wandb.log({
        "test_accuracy": (100 * test_correct / len(test_loader)) 
            })

    print('Test Accuracy: %.1f'%(100 * test_correct / len(test_loader)))


# Save the model
#model.to_onnx()
wandb.save("777_training.onnx")


# metrics should be added
#for batch in dataloader:
#  metrics = model.training_step()
# # log metrics inside your training loop to visualize model performance
#  wandb.log(metrics)


