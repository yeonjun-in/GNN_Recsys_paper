import numpy as np
import pandas as pd
import os, sys, pickle

from collections import defaultdict
import networkx as nx
from copy import deepcopy
from operator import itemgetter 

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import dgl
from dgl.data import FB15k237Dataset

from datetime import datetime
import matplotlib.pyplot as plt

import os, random, warnings, math, gc
warnings.filterwarnings('ignore')

from dataset import KgDataset, prepare_neg_entity
from model import TransE

head = torch.load('./data/KG_head.pt').numpy()
tail = torch.load('./data/KG_tail.pt').numpy()
label = torch.load('./data/KG_label.pt').numpy()

class Config:
    learning_rate = 0.001
    weight_decay = 0.001
    batch_size = 1024
    embed_dim = 36
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    neg_sample = 10 # h'와 t' 각각 count
    gamma = 1
    epochs = 50

config = Config()

head_neg, tail_neg = prepare_neg_entity(head, tail, label)

dataset = KgDataset(head, tail, label, config.neg_sample, head_neg, tail_neg)
dataloader = DataLoader(dataset, config.batch_size, drop_last=False, shuffle=True)
model = TransE(num_entity=max(head)+1,
               num_label=max(label)+1,
               embed_dim=config.embed_dim,
               gamma=config.gamma,
               config=config)
model = model.to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

history = defaultdict(list)
for epoch in range(config.epochs):
    losses = []
    model.train()
    for batch_data in dataloader:
        optimizer.zero_grad()
        batch_data = {k:v.to(config.device) for k,v in batch_data.items()}
        loss = model(batch_data)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # if (epoch+1) % 10 == 0 or epoch==0 or (epoch+1)==config.epochs:
    print(f'EPOCH {epoch+1} : Loss {np.mean(losses):.1f}')
    history['train'].append(np.mean(losses))

plt.plot(history['train'])