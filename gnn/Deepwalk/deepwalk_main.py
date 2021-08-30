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

from dataset import WalkDatset
from deepwalk import DeepWalk
from utils import make_graph_data

from datetime import datetime
import matplotlib.pyplot as plt

import os, random, warnings, math
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

class Config:
    learning_rate = 0.001
    weight_decay = 0.01
    epochs = 10
    seed = 1995
    embed_dim = 30
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    batch_size = 128
    walk_len = 20
    num_walk_per_node = 100
    window_size = 5

config = Config()

graph = make_graph_data('ind.citeseer.graph', weighted=False, num_type_node=False)[0]
config.tree_depth = int(np.ceil(np.log2(graph.number_of_nodes())))

dataset = WalkDatset(graph, config.walk_len, config.window_size)
dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=False, shuffle=True)

model = DeepWalk(graph.nodes(), config.embed_dim, config.device)
model = model.to(config.device)

optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

history = defaultdict(list)
start = datetime.now()
for epoch in range(config.num_walk_per_node):
    losses = []
    model.train()
    for batch_data in dataloader:
        optimizer.zero_grad()
        target, context = batch_data['target'].to(config.device), batch_data['context'].to(config.device)
        loss = model(target, context)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0 or epoch==0 or (epoch+1)==config.num_walk_per_node:
        print(f'EPOCH {epoch+1} : Loss', np.mean(losses))
    history['train'].append(np.mean(losses))

end = datetime.now()
print(end-start)
plt.plot(history['train'])

