


import torch
import os
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pytorch_warmup as warmup
import pickle
import numpy as np
import math
import torchvision
torch.cuda.set_device(1)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_size = 32 * 32 * 3
batch_size = 512
num_epochs = 100
learning_rate = 0.01
hidden_size = 512
number_H =5
random_seed = 42

epoch_interval = 1


address =''
name = 'lr_'+ 'Radam'+\
        '_learning_rate:'+ str(learning_rate) +\
        'batch_size:'+str(batch_size)+\
        '_num_epochs:'+ str(num_epochs) +\
        '_hidden_size:' +str(hidden_size) + \
        '_number_H:' + str(number_H) 

address_1 = address+name+'_cos_sim.txt'
address_2 = address+name+'_acc.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')

def seed_torch(seed=random_seed):
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

def cos_similarity_matrix_row(matrix):
    num_rows = matrix.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))
    for i in range(num_rows):
        for j in range(i, num_rows):
            similarity_matrix[i, j] = 1 - cosine(matrix[i], matrix[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def cos_similarity_matrix_column(matrix):
    num_column = matrix.shape[1]
    similarity_matrix = np.zeros((num_column, num_column))
    for i in range(num_column):
        for j in range(i, num_column):
            similarity_matrix[i, j] = 1 - cosine(matrix[:,i], matrix[:,j])
            similarity_matrix[j, i] = similarity_matrix[i, j]
    return abs(similarity_matrix)
def Mean(matrix):
    number = matrix.shape[0]
    matrix_mean = matrix.mean()
    mean_out = abs((matrix_mean - (1/number))*(number/(number-1)))
    return mean_out
def Gram_matrix_row(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix,matrix_transpose)
    return Gram_matrix
def Gram_matrix_column(matrix):
    matrix_transpose = np.transpose(matrix)
    Gram_matrix = np.dot(matrix_transpose,matrix)
    return Gram_matrix

def cos_similarity(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"{name}"+'  ')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    print(mean_cos_sim_row)
    file1.writelines(str(mean_cos_sim_row)+'  ')
    print('='*50)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear = nn.Linear(input_size, hidden)
        self.r = nn.ReLU()
        self.hidden = hidden
        self.linearH = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(number_H)])
        self.out = nn.Linear(hidden, num_classes)
        self.norm = nn.BatchNorm1d(hidden)
    
    def forward(self, x):
        
        x = x.view(-1, input_size)
        x = self.linear(x)
        x = self.r(x)
        
        for i in  range(number_H):
            x = self.linearH[i](x)
            x = self.r(x) 

        out = self.out(x)
        return out


if torch.cuda.is_available():
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10).cuda()
else:
    model = feedforward_neural_network(input_size = input_size, hidden = hidden_size, num_classes = 10)
criterion = nn.CrossEntropyLoss()



optimizer = torch.optim.RAdam(model.parameters(), lr = learning_rate)
optimizer.zero_grad()


for name, param in model.named_parameters():
    print(name)

file1.writelines('epoch:'+str(0)+'  ')
for name, param in model.named_parameters():

    if name == 'linearH.0.weight':
        cos_similarity(name)
    if name == 'linearH.1.weight':
        cos_similarity(name)
    if name == 'linearH.2.weight':
        cos_similarity(name)
    if name == 'linearH.3.weight':
        cos_similarity(name)        
    if name == 'linearH.4.weight':
        cos_similarity(name)    
        file1.writelines('\n') 

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i,(features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        targets = Variable(targets).cuda()
        probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs= model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

     


        loss_out=round(loss.item(),4)
        if (i+1) % 40 == 0:
            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                  (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,loss.item()))
    text_accuracy = round(compute_accuracy(model, test_loader).item(),4)
    print(str(text_accuracy)+'  '+str(loss_out))
    file2.writelines(str(text_accuracy)+'  '+str(loss_out)+'\n')
    if (epoch+1) % epoch_interval == 0:
        file1.writelines('epoch:'+str(epoch)+'  ')
        for name, param in model.named_parameters():   
            if name == 'linearH.0.weight':
                cos_similarity(name)
            if name == 'linearH.1.weight':
                cos_similarity(name)
            if name == 'linearH.2.weight':
                cos_similarity(name)
            if name == 'linearH.3.weight':
                cos_similarity(name)        
            if name == 'linearH.4.weight':
                cos_similarity(name)
                file1.writelines('\n')

file1.close() 
file2.close()






