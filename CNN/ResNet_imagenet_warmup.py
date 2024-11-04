import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.distance import cosine
import time 
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup


data_dir = 
batch_size = 864
num_epochs = 10
learning_rate = 0.001
num_classes = 1000  

address ='/home/sda/luzhixing/warmup/CNN/result/imagenet/'
Name = 'warmup_'+ 'adam'+\
        '_batch_size:'+str(batch_size)+\
        '_learning_rate:'+str(learning_rate) +\
          '_num_epochs:'+ str(num_epochs)
        
address_1 = address+Name+'_cos_sim.txt'
address_2 = address+Name+'_acc.txt'
file1 = open(address_1,'w')
file2 = open(address_2,'w')

def seed_torch(seed=42):
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

def cos_similarity_conv(name,param):
    file1.writelines(f" {name}"+'  ')
    cos_sim_row = cos_sim_column = 0.0
    param = param.cpu()
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


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



train_dataset = datasets.ImageNet(root=data_dir+'/train', split='train', transform=data_transforms['train'])
val_dataset = datasets.ImageNet(root=data_dir+'/val', split='val', transform=data_transforms['val'])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


model = models.resnet18(pretrained=True)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 5, num_training_steps =num_epochs )



for name, param in model.named_parameters():
    if name == 'layer4.0.conv2.weight':
        print(f'{name}: {param.size()}')
        file1.writelines('epoch:'+str(0)+'  ')
        cos_similarity_conv(name,param)
        file1.writelines('\n') 

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start_time = time.time() 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        epoch_start_time = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

         
            for i, (inputs, labels) in (enumerate(train_loader) if phase == 'train' else enumerate(val_loader)):
                inputs = inputs.to(device)
                labels = labels.to(device)


                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
     
                        if (i+1) % 500 == 0:
                            print('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                                    (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,loss.item()))
                            file2.writelines('Epoch: [%d/%d],Step:[%d/%d],Loss:%.4f ' % 
                                    (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,loss.item())+'\n')

                            file1.writelines('epoch:'+str(epoch+1)+'  ')
                            for name, param in model.named_parameters():
                                if name == 'layer4.0.conv2.weight':
                                    print('---------------------------------------------')
                                    print(f'{name}: {param.size()}')
                                    cos_similarity_conv(name,param)
                                    file1.writelines('\n') 


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(train_dataset if phase == 'train' else val_dataset)
            epoch_acc = running_corrects.double() / len(train_dataset if phase == 'train' else val_dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            file2.writelines(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'+'\n')


        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time

        remaining_time = elapsed_time / (epoch + 1) * (num_epochs - (epoch + 1))
        print(f'Epoch Time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
        print(f'Elapsed Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s')
        print(f'Estimated Remaining Time: {remaining_time // 60:.0f}m {remaining_time % 60:.0f}s')
    
    
    total_training_time = time.time() - start_time
    print(f'Training complete in {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s')

    return model

model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

torch.save(model.state_dict(), 'resnet18_imagenet.pth')

file1.close()
file2.close()
print("Training finished.")