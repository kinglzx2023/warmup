#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
torch.cuda.set_device(0)
import torchvision.transforms as transforms


# In[2]:


input_size = 784
batch_size = 512
num_epochs = 1000
learning_rate = 0.001
hidden_size = 50
number_H =3
probability = 0.5


# In[3]:


address = './'+'original_'+'{}'.format(batch_size)+'_{}'.format(hidden_size)+'_'+'{}'.format(learning_rate)+'_'+'{}'.format(number_H)
address1 = address+'_'+'{}'.format(num_epochs)+'_'+'parameters.txt'
file1 = open(address1,'w')


# In[4]:


def seed_torch(seed=42):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False   
seed_torch()


# In[5]:


train_datasets = dsets.MNIST(root = './Datasets', train = True, download = True, transform = transforms.ToTensor())
test_datasets = dsets.MNIST(root = './Datasets', train = False, download = True, transform = transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset = train_datasets, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_datasets, batch_size = batch_size, shuffle = False)


# In[6]:


class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.dropout = nn.Dropout(probability)
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
    def forward(self, x):
        tensor = torch.tensor((),dtype = torch.float32)
        #x = self.dropout(x)
        x = self.linear(x)
        x = self.r(x)
        
        for i in  range(number_H):
            #x = self.dropout(x)
            x = self.linearH[i](x)
            x = self.r(x)
        out = self.out(x)
        return out


# In[9]:


if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)

    
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()


# In[10]:



def distance(parameters):
    distance = []
    for param in parameters:
        dis = math.sqrt(torch.sum(torch.mul(param,param)).item())
        distance.append(round(dis,4))
    return distance

def parameters_mean(parameters):
    mean = []
    for param in parameters:
        mean_unit = torch.mean(param).item()
        mean.append(round(mean_unit,4))
    return mean
def parameters_var(parameters):
    var = []
    for param in parameters:
        var_unit = torch.var(param).item()
        var.append(round(var_unit,4))
    return var

def weights_distribution(parameters):   
    weights = []
    for param in parameters:
        weight_unit = param.cpu().detach().numpy()
        weight_unit = np.round(weight_unit,6)
        weight_unit = weight_unit.tolist()
        weights.append(weight_unit)
    return weights
def round_2(list):
    List_round_2=[]
    for unit in list:
        List_round_2.append(round(unit,6))
    return List_round_2


# In[15]:


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        #features = features.to(device)
        features = Variable(features.view(-1, 28*28)).cuda()
        
        #targets = targets.to(device)
        targets = Variable(targets).cuda()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


# In[16]:


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28*28)).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)
        outputs= model(images)

        optimizer.zero_grad() 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        weight_distribution = weights_distribution(model.parameters())
        loss_out=round(loss.item(),4)

        if (i+1) % 40 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_datasets)//batch_size,loss.item()))
            
    text_accuracy = round(compute_accuracy(model, test_loader).item(),4)
    weight_output = str(round_2(weight_distribution[2][0]))
    bias_output = str(round(weight_distribution[3][0],6))
    print(str(text_accuracy)+'  '+str(loss_out))
    file1.writelines(str(text_accuracy)+'  '+str(loss_out)+'  '+weight_output+'   '+bias_output+'\n')      
file1.close()




