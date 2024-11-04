
import torch
import os
import torchvision
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
import pickle
import numpy as np
import math
torch.cuda.set_device(0)
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine



Batch_size = 512
num_epochs = 100
learning_rate = 0.00001
random_seed = 42
epoch_interval = 1


Momentum = 0.0
Alpha = 0.0
beta1 = 0.0
beta2 = 0.0

address =''

name = 'lr_'+'batch_size:'+str(Batch_size)+'_learning_rate:'+\
         str(learning_rate)  + '_random_seed :'+ str(random_seed )+\
          '_num_epochs:'+ str(num_epochs)+\
         '_optimizer:'+ 'Radam'

address_1 = address+name+'_cos_sim.txt'
address_2 = address+name+'_acc.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')



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

def cos_similarity_conv(name):
    #print(f"Parameter name: {name}")
    file1.writelines(f" {name}"+'  ')
    #print(f"Parameter value: {param.data.size()}")
    cos_sim_row = cos_sim_column = 0.0
    for i in range(param.size(2)):
        for j in range(param.size(3)):
            param_i_j = param.data[:,:,i,j]
            cos_sim_row = cos_sim_row + cos_similarity_matrix_row(param_i_j .cpu().data)
            cos_sim_column =cos_sim_column + cos_similarity_matrix_column(param_i_j .cpu().data) 
    
    cos_sim_row = Mean(cos_sim_row) /(param.size(2)*param.size(3))
    cos_sim_column = Mean(cos_sim_column) /(param.size(2)*param.size(3))
    mean_cos_sim_row = round(cos_sim_row,6)
    mean_cos_sim_column= round(cos_sim_column,6)
    
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+'  '+ str(mean_cos_sim_column)+'  ')

def cos_similarity_ffn(name):
    print(f"Parameter name: {name}")
    file1.writelines(f"{name}"+'\n')
    print(f"Parameter value: {param.data.size()}")  
    cos_sim_row = cos_similarity_matrix_row(param.cpu().data)
    cos_sim_column = cos_similarity_matrix_column(param.cpu().data) 
    mean_cos_sim_row = round(Mean(cos_sim_row),6)
    mean_cos_sim_column= round(Mean(cos_sim_column),6)
    print(mean_cos_sim_row, mean_cos_sim_column)
    file1.writelines(str(mean_cos_sim_row)+','+ str(mean_cos_sim_column)+'\n')
    file1.writelines('='*50+'\n')
    print('='*50)


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


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.activation_f = nn.ReLU() #GELU,Tanh
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
      
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
   
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
     

        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation_f(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.activation_f(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.activation_f(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation_f(x)

        x = self.fc2(x)

        return x


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainset = torchvision.datasets.CIFAR10(root='/home', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.RAdam(model.parameters(), lr = learning_rate)


for name, param in model.named_parameters():
    print(f'{name}: {param.size()}')



for name, param in model.named_parameters():

    if name == 'conv1.weight':
        cos_similarity_conv(name)
    if name == 'conv2.weight':
        cos_similarity_conv(name)
    if name == 'conv3.weight':
        cos_similarity_conv(name)
        file1.writelines('\n') 

    


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = total_loss / len(trainloader)
    train_accuracy = correct / total


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    file2.writelines(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}'+'\n')

    if (epoch+1) % epoch_interval == 0:
        file1.writelines('epoch:'+str(epoch)+'  ')
        for name, param in model.named_parameters():
            if name == 'conv1.weight':
                cos_similarity_conv(name)
            if name == 'conv2.weight':
                cos_similarity_conv(name)
            if name == 'conv3.weight':
                cos_similarity_conv(name)
                file1.writelines('\n')
        
file1.close()
file2.close()
print("Training finished.")


