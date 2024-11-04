#!/usr/bin/env python
# coding: utf-8

# In[66]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', "-a 'Sebastian Raschka' -v -p torch")
import torch
import numpy as np
import torch.nn.functional as F
from torchtext.legacy import data
#from torchtext.legacy.data import Field
from torchtext.legacy import datasets
import time
import random
import spacy
torch.cuda.set_device(1)
from torch.autograd import Variable

torch.backends.cudnn.deterministic = True


# ## General Settings

# In[67]:


RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
VOCABULARY_SIZE = 20000
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 1
drop_prob = 0.5


# In[68]:


address = '/home/sda/luzhixing/paper_2/deeplearning-models-master/pytorch_ipynb/rnn/result/'+'{}'.format(BATCH_SIZE)+'_{}'.format(HIDDEN_DIM)+'_'+'{}'.format(LEARNING_RATE)+'_'+'{}'.format(NUM_EPOCHS)
address1 = address+'_'+'w_hi.txt'
address2 = address+'_'+'w_hf.txt'
address3 = address+'_'+'w_hg.txt'
address4 = address+'_'+'w_ho.txt'
address5 = address+'_'+'FC.txt'
file1 = open(address1,'w')
file2 = open(address2,'w')
file3 = open(address3,'w')
file4 = open(address4,'w')
file5 = open(address5,'w')


# In[69]:


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


# In[70]:


TEXT = data.Field(tokenize='spacy',
                  include_lengths=True) # necessary for packed_padded_sequence
#TEXT = data.Field(include_lengths=True) # necessary for packed_padded_sequence
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
train_data, valid_data = train_data.split(random_state=random.seed(RANDOM_SEED),
                                          split_ratio=0.8)

print(f'Num Train: {len(train_data)}')
print(f'Num Valid: {len(valid_data)}')
print(f'Num Test: {len(test_data)}')


# In[71]:


TEXT.build_vocab(train_data,
                 max_size=VOCABULARY_SIZE,
                 vectors='glove.6B.100d',
                 unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(f'Vocabulary size: {len(TEXT.vocab)}')
print(f'Number of classes: {len(LABEL.vocab)}')


# In[72]:


train_loader, valid_loader, test_loader = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    sort_within_batch=True, # necessary for packed_padded_sequence
    device=DEVICE)


# In[73]:


print('Train')
for batch in train_loader:
    print(f'Text matrix size: {batch.text[0].size()}')
    print(f'Target vector size: {batch.label.size()}')
    break
    
print('\nValid:')
for batch in valid_loader:
    print(f'Text matrix size: {batch.text[0].size()}')
    print(f'Target vector size: {batch.label.size()}')
    break
    
print('\nTest:')
for batch in test_loader:
    print(f'Text matrix size: {batch.text[0].size()}')
    print(f'Target vector size: {batch.label.size()}')
    break


# ## Model

# In[74]:


import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, text, text_length):
        embedded = self.embedding(text)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_length.to('cpu'))
        packed_output, (hidden, cell) = self.rnn(packed)
        
        output = self.dropout(hidden.squeeze(0))
        return self.fc(output).view(-1)


# In[75]:


INPUT_DIM = len(TEXT.vocab)
print(INPUT_DIM)
torch.manual_seed(RANDOM_SEED)
model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ## Training

# In[77]:


def compute_binary_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            text, text_lengths = batch_data.text
            logits = model(text, text_lengths)
            predicted_labels = (torch.sigmoid(logits) > 0.5).long()
            num_examples += batch_data.label.size(0)
            correct_pred += (predicted_labels == batch_data.label.long()).sum()
        return correct_pred.float()/num_examples * 100


# In[78]:


start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
             
        text, text_lengths = batch_data.text
        logits = model(text, text_lengths)
        cost = F.binary_cross_entropy_with_logits(logits, batch_data.label)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        if not batch_idx % 50:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                   f'Cost: {cost:.4f}')
            weight_distribution = weights_distribution(model.parameters())
            
            weight_output_hi = str(round_2(weight_distribution[2][0]))
            bias_output_hi = str(round(weight_distribution[3][0],6))

            
            weight_output_hf = str(round_2(weight_distribution[2][256]))
            bias_output_hf = str(round(weight_distribution[3][256],6))
            
            weight_output_hg = str(round_2(weight_distribution[2][512]))
            bias_output_hg = str(round(weight_distribution[3][512],6))
            
            weight_output_ho = str(round_2(weight_distribution[2][768]))
            bias_output_ho = str(round(weight_distribution[3][768],6))
            
            weight_output_FC = str(round_2(weight_distribution[5][0]))
            bias_output_FC = str(round(weight_distribution[6][0],6))

            file1.writelines(weight_output_hi+'   '+bias_output_hi+'\n')
            file2.writelines(weight_output_hf+'   '+bias_output_hf+'\n')
            file3.writelines(weight_output_hg+'   '+bias_output_hg+'\n')
            file4.writelines(weight_output_ho+'   '+bias_output_ho+'\n')
            file5.writelines(weight_output_FC+'   '+bias_output_FC+'\n')            

    with torch.set_grad_enabled(False):
        print(f'training accuracy: '
              f'{compute_binary_accuracy(model, train_loader, DEVICE):.2f}%'
              f'\nvalid accuracy: '
              f'{compute_binary_accuracy(model, valid_loader, DEVICE):.2f}%')
        
    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
    
print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {compute_binary_accuracy(model, test_loader, DEVICE):.2f}%')
file1.close()
file2.close()
file3.close()
file4.close()
file5.close()


# In[ ]:




