import numpy as np
import pandas as pd
import os, sys, pickle, random, warnings, math

from collections import defaultdict
import networkx as nx

from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils import make_graph_data, SamplingAliasMethod

from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

class WalkDatset(Dataset):
    def __init__(self, graph, walk_length, window_size):
        self.graph = graph
        self.walk_length = walk_length
        self.window_size = window_size
        self.nodes = list(graph.nodes())

    def __len__(self):
        return len(self.nodes)

    def random_walk(self, graph, node, walk_length):
        src = node
        walk = [src]
        for i in range(walk_length):
            dst = list(dict(graph[src]).keys())
            src = np.random.choice(dst)
            walk.append(src)
        return walk
    
    def __getitem__(self, idx):
        node = self.nodes[idx]
        walk = self.random_walk(self.graph, node, self.walk_length)
        target_idx, context_idx = [],[]
        for i, target in enumerate(walk):
            left_window = walk[max(i-self.window_size, 0):i]
            right_window = walk[i+1:i+1+self.window_size]
            contexts = left_window + right_window
            
            target_idx.extend([target] * len(contexts))
            context_idx.extend(contexts)
        
        return {'target':torch.LongTensor(target_idx), 'context':torch.LongTensor(context_idx)}