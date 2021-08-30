import numpy as np
import pandas as pd

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
import matplotlib.pyplot as plt

import os, random, warnings, math, gc
warnings.filterwarnings('ignore')

import dgl
from dgl.data import CoraGraphDataset

from dataset import GCNDataset, get_A_mat
from model import GCNLayer, GCN
from utils import seed_everything

class Config:
    learning_rate = 0.01
    weight_decay = 5e-4
    hidden_dim = 16
    epochs = 200
    early_stopping_round = None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 1995

config = Config()
dataset = CoraGraphDataset()
graph = dataset[0]
config.batch_size = graph.num_nodes()
config.input_dim = graph.ndata['feat'].shape[1]
config.output_dim = graph.ndata['label'].unique().shape[0]

seed_everything(config.seed)
train_set = GCNDataset(graph, True)
valid_set = GCNDataset(graph, False)
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)

A = get_A_mat(graph, config)
model = GCN(config)
model = model.to(config.device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
loss_fn = nn.CrossEntropyLoss()
history = defaultdict(list)

start = datetime.now()
best_loss, early_step, best_epoch = 0, 0, 0
for epoch in range(config.epochs):
    model.train()
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = {k:v.to(config.device) for k,v in batch_data.items()}
        output, true = model(batch_data, A)
        acc_tr = torch.sum(true == torch.argmax(output, axis=1)) / len(true)
        loss = loss_fn(output, true)
        loss.backward()
        optimizer.step()

    history['train_loss'].append(loss.item())
    history['train_acc'].append(acc_tr)

    model.eval()
    with torch.no_grad():
        for batch_data in valid_loader:
            batch_data = {k:v.to(config.device) for k,v in batch_data.items()}
            output, true = model(batch_data, A)
            acc = torch.sum(true == torch.argmax(output, axis=1)) / len(true)
            loss = loss_fn(output, true)

    history['valid_loss'].append(loss.item())
    history['valid_acc'].append(acc)

    if epoch == 0 or epoch == config.epochs-1 or (epoch+1)%10 == 0:
        print(f'EPOCH {epoch+1} : TRAINING loss {history["train_loss"][-1]:.3f}, TRAINING ACC {history["train_acc"][-1]:.3f}, VALID loss {history["valid_loss"][-1]:.3f}, VALID ACC {history["valid_acc"][-1]:.3f}')
    
    if history['valid_acc'][-1] > best_loss:
        best_loss = history['valid_acc'][-1]
        best_epoch = epoch

    elif(config.early_stopping_round is not None):
        
        early_step += 1
        if (early_step >= config.early_stopping_round):
            break
end = datetime.now()
print(end-start)
print(f'At EPOCH {best_epoch + 1}, We have Best Acc {best_loss}')