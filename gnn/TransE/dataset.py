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

class KgDataset(Dataset):
    def __init__(self, head, tail, label, neg_sample_k, head_neg, tail_neg):
        super(KgDataset, self).__init__()
        self.head = head
        self.tail = tail
        self.label = label
        self.neg_sample_k = neg_sample_k
        self.head_neg = head_neg
        self.tail_neg = tail_neg

    def __len__(self):
        return len(self.head)

    def __getitem__(self, idx):
        head, tail, label = self.head[idx], self.tail[idx], self.label[idx]
        tail_prime = np.random.choice(self.head_neg[head], self.neg_sample_k)
        head_prime = np.random.choice(self.tail_neg[tail], self.neg_sample_k)

        return {'head':head, 'label':label, 'tail':tail, 'tail_p':tail_prime, 'head_p':head_prime}

def prepare_neg_entity(head, tail, label):
    adj_mat = np.zeros((np.max(head)+1, np.max(head)+1))
    for h,t,l in zip(head, tail, label):
        adj_mat[h,t] = l

    idx = torch.arange(np.max(head)+1)
    head_neg = {h:idx[~adj_mat[h].astype(bool)] for h in np.unique(head)}
    tail_neg = {t:idx[~adj_mat[:, t].astype(bool)] for t in np.unique(tail)}

    del adj_mat; gc.collect()
    return head_neg, tail_neg