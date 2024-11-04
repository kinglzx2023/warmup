#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[2]:


import os
import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


# In[3]:


from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss
from collections import OrderedDict
from helper_data import get_dataloaders_cifar10
import json
import subprocess
import sys
import xml.etree.ElementTree


# In[4]:


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# In[5]:


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)


# In[6]:


# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
NUM_EPOCHS = 100
# Architecture
NUM_CLASSES = 10
# Other
DEVICE = "cuda:0"
set_all_seeds(RANDOM_SEED)


# #### Import utility functions

# In[8]:


address = './'
address1 = address+'_'+'layer-1.txt'
address2 = address+'_'+'layer-2.txt'
address3 = address+'_'+'layer-3.txt'
address4 = address+'_'+'layer-4.txt'
address5 = address+'_'+'layer-5.txt'
address6 = address+'_'+'layer-6.txt'
address7 = address+'_'+'layer-7.txt'
address8 = address+'_'+'layer-8.txt'
file1 = open(address1,'w')
file2 = open(address2,'w')
file3 = open(address3,'w')
file4 = open(address4,'w')
file5 = open(address5,'w')
file6 = open(address6,'w')
file7 = open(address7,'w')
file8 = open(address8,'w')


# In[9]:


def distance(parameters):
    distance = []
    for param in parameters:
        dis = math.sqrt(torch.sum(torch.mul(param,param)).item())
        distance.append(round(dis,4))
    return distance
#计算参数的均值
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
def reduce_dim(list):
    List_init=[]
    for Q in list:
        for K in Q:
            for V in K:
                List_init.append(V)
    return List_init


# ## Dataset

# In[10]:


set_all_seeds(RANDOM_SEED)
train_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                       transforms.RandomCrop((64, 64)),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize((70, 70)),
                                      transforms.CenterCrop((64, 64)),
                                      transforms.ToTensor()])
train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
    batch_size=BATCH_SIZE, 
    num_workers=2, 
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    validation_fraction=0.1)


# In[11]:


# Checking the dataset
print('Training Set:\n')
for images, labels in train_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break
# Checking the dataset
print('\nValidation Set:')
for images, labels in valid_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break
# Checking the dataset
print('\nTesting Set:')
for images, labels in train_loader:  
    print('Image batch dimensions:', images.size())
    print('Image label dimensions:', labels.size())
    print(labels[:10])
    break


# ## Model

# In[12]:


class AlexNet(nn.Module):

    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.AvgPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.AvgPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #nn.AvgPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits


# In[13]:


torch.manual_seed(RANDOM_SEED)
model = AlexNet(NUM_CLASSES)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  


# In[14]:


for item in model.parameters():
    print(item.size())


# In[15]:


def train_classifier_simple_v1(num_epochs, model, optimizer, device, 
                               train_loader, valid_loader=None, 
                               loss_fn=None, logging_interval=100, 
                               skip_epoch_stats=False):
    
    log_dict = {'train_loss_per_batch': [],
                'train_acc_per_epoch': [],
                'train_loss_per_epoch': [],
                'valid_acc_per_epoch': [],
                'valid_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))
                weight_distribution = weights_distribution(model.parameters())
                
                weight_output_1 = str(round_2(reduce_dim(weight_distribution[0][10])))
                bias_output_1 = str(round(weight_distribution[1][10],6))
                
                weight_output_2 = str(round_2(reduce_dim(weight_distribution[2][10])))
                bias_output_2 = str(round(weight_distribution[3][10],6))
                
                weight_output_3 = str(round_2(reduce_dim(weight_distribution[4][10])))
                bias_output_3 = str(round(weight_distribution[5][10],6))
                
                weight_output_4 = str(round_2(reduce_dim(weight_distribution[6][10])))
                bias_output_4 = str(round(weight_distribution[7][10],6))
                
                weight_output_5 = str(round_2(reduce_dim(weight_distribution[8][10])))
                bias_output_5 = str(round(weight_distribution[9][10],6))
                
                weight_output_6 = str(round_2(weight_distribution[10][10]))
                bias_output_6 = str(round(weight_distribution[11][10],6))
                
                weight_output_7 = str(round_2(weight_distribution[12][10]))
                bias_output_7 = str(round(weight_distribution[13][10],6))
                
                weight_output_8 = str(round_2(weight_distribution[14][5]))
                bias_output_8 = str(round(weight_distribution[15][5],6))
                
                file1.writelines(weight_output_1+'   '+bias_output_1+'\n')
                file2.writelines(weight_output_2+'   '+bias_output_2+'\n')
                file3.writelines(weight_output_3+'   '+bias_output_3+'\n')
                file4.writelines(weight_output_4+'   '+bias_output_4+'\n')
                file5.writelines(weight_output_5+'   '+bias_output_5+'\n')
                file6.writelines(weight_output_6+'   '+bias_output_6+'\n')
                file7.writelines(weight_output_7+'   '+bias_output_7+'\n')
                file8.writelines(weight_output_8+'   '+bias_output_8+'\n')
                


        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_acc = compute_accuracy(model, train_loader, device)
                train_loss = compute_epoch_loss(model, train_loader, device)
                print('***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f' % (
                      epoch+1, num_epochs, train_acc, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())
                log_dict['train_acc_per_epoch'].append(train_acc.item())

                if valid_loader is not None:
                    valid_acc = compute_accuracy(model, valid_loader, device)
                    valid_loss = compute_epoch_loss(model, valid_loader, device)
                    print('***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f' % (
                          epoch+1, num_epochs, valid_acc, valid_loss))
                    log_dict['valid_loss_per_epoch'].append(valid_loss.item())
                    log_dict['valid_acc_per_epoch'].append(valid_acc.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    file1.close()
    file2.close()
    file3.close()
    file4.close()
    file5.close()
    file6.close()
    file7.close()
    file8.close()
    return log_dict


# ## Training

# In[16]:


log_dict = train_classifier_simple_v1(num_epochs=NUM_EPOCHS, model=model, 
                                      optimizer=optimizer, device=DEVICE, 
                                      train_loader=train_loader, valid_loader=valid_loader, 
                                      logging_interval=50)
#file1.close()


# ## Evaluation
