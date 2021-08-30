import numpy as np
import pandas as pd
import os, sys, pickle

from collections import defaultdict
import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim


from datetime import datetime
import matplotlib.pyplot as plt

import os, random, warnings, math
warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

class DeepWalk(nn.Module):
    def __init__(self, nodes, emb_dim, device):
        super(DeepWalk, self).__init__()
        self.embed = nn.Embedding(len(nodes), emb_dim)
        self.all_node = torch.LongTensor(list(nodes)).to(device)
        # torch.nn.init.normal_(self.embed.weight, mean=0, std=0.02)        # to

    def hier_softmax(self,):
        return 

    def forward(self, target, context):
        batch_size, _ = target.size()
        # print(_)
        embed_t = self.embed(target)
        embed_c = self.embed(context)
        embed_all = self.embed(self.all_node).T.unsqueeze(0).repeat(batch_size, 1, 1)
        score = torch.exp(torch.sum(torch.mul(embed_t, embed_c), axis=2))
        scale = torch.sum(torch.exp(torch.bmm(embed_t, embed_all)), axis=2)
        loss = -torch.log(score / scale)
        # print(loss.shape)

        return torch.mean(torch.mean(loss, axis=1))