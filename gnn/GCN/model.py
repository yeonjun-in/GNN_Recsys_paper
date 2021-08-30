import torch
import torch.nn as nn
import torch.nn.functional as F
import os, random, warnings, math, gc
warnings.filterwarnings('ignore')


class GCNLayer(nn.Module):
    def __init__(self, input, output, dropout):
        super(GCNLayer, self).__init__()
        self.input = input
        self.output = output
        self.W = nn.Linear(input, output)
        self.dropout = nn.Dropout(dropout)
        # torch.nn.init.uniform_(self.W.weight, -1/math.sqrt(output), 1/math.sqrt(output))
        torch.nn.init.uniform_(self.W.weight)        
    
    def forward(self, x, adj):
        output = torch.spmm(adj, x)
        output = self.dropout(output)
        output = self.W(output)
        return output

class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(config.input_dim, config.hidden_dim, dropout=0.1) 
        self.gcn2 = GCNLayer(config.hidden_dim, config.output_dim, dropout=0.1) 
        
    def forward(self, batch_data, A):
        label, data, mask = batch_data['y'], batch_data['x'], batch_data['mask']
        data = F.relu(self.gcn1(data, A))
        data = self.gcn2(data, A)
        return data[mask], label[mask]

# 처음에 이렇게 했다가 생각해보니 layer 늘어나면 안좋을 것 같아서, GCNLayer class 추가
# class GCN(nn.Module):
#     def __init__(self, config):
#         super(GCN, self).__init__()
#         self.W0 = nn.Linear(config.input_dim, config.hidden_dim) 
#         self.W1 = nn.Linear(config.hidden_dim, config.output_dim) 
#         self.dropout1 = nn.Dropout(0.1)
#         self.dropout2 = nn.Dropout(0.1)

#         # torch.nn.init.uniform_(self.W0.weight, -1/math.sqrt(config.hidden_dim), 1/math.sqrt(config.hidden_dim))
#         # torch.nn.init.uniform_(self.W1.weight, -1/math.sqrt(config.output_dim), 1/math.sqrt(config.output_dim))

#         torch.nn.init.uniform_(self.W0.weight)
#         torch.nn.init.uniform_(self.W1.weight)
    
#     def forward(self, batch_data, A):
#         label, data, mask = batch_data['y'], batch_data['x'], batch_data['mask']
#         data = torch.spmm(A, data)
#         data = self.dropout1(data)
#         data = F.relu(self.W0(data))

#         data = torch.spmm(A, data)
#         data = self.dropout2(data)
#         data = self.W1(data)
#         return data[mask], label[mask]