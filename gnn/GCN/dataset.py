import numpy as np
import pandas as pd
import os, sys, pickle

from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime

import os, random, warnings, math, gc
warnings.filterwarnings('ignore')

import dgl
from dgl.data import CoraGraphDataset

class GCNDataset(Dataset):
    def __init__(self, graph, is_train):
        super(GCNDataset, self).__init__()
        self.graph = graph
        self.mask = graph.ndata['train_mask'] if is_train else graph.ndata['test_mask']
        self.label = graph.ndata['label']
        self.node = graph.nodes()
        self.feat = graph.ndata['feat'].float()

    def __len__(self):
        return self.graph.num_nodes()

    def __getitem__(self, idx):
        return {
            'node': self.node[idx],
            'y': self.label[idx],
            'mask': self.mask[idx],
            'x': self.feat[idx]
        }

def get_A_mat(graph, config):
    A = np.zeros((graph.num_nodes(), graph.num_nodes()))
    for src, dst in zip(graph.edges()[0].numpy(), graph.edges()[1].numpy()):
        A[src, dst] += 1
    A = A + np.identity(graph.num_nodes())
    D = np.sum(A, axis=1)
    D = np.diag(np.power(D, -0.5))
    Ahat = np.dot(D, A).dot(D)
    return torch.tensor(Ahat).float().to(config.device)