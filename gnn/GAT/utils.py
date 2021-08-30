import numpy as np
import torch 
import torch.nn as nn
import dgl
from dgl.data import CoraGraphDataset

import random, os

def load_data():

    graph = CoraGraphDataset()[0]
    train_mask = ~(graph.ndata['test_mask'] |  graph.ndata['val_mask'])
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    feat = graph.ndata['feat']
    label = graph.ndata['label']
    n_nodes = graph.num_nodes()
    edges = graph.edges()
    adj = np.zeros((n_nodes, n_nodes))
    for src, dst in zip(edges[0].numpy(), edges[1].numpy()):
        adj[src, dst] += 1
    
    return train_mask, val_mask, test_mask, feat, label, torch.LongTensor(adj)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True